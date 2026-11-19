import streamlit as st
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# Page configuration
# =========================================================

st.set_page_config(
    page_title="RideOps L1 Escalation Assistant",
    page_icon="🚕",
    layout="wide",
)


# =========================================================
# Load data
# =========================================================

DATA_PATH = "data/rideops_support_tickets.csv"

df = pd.read_csv(DATA_PATH)

MAX_SIMILAR_TICKETS = 20
MIN_SIMILARITY = 0.20


# =========================================================
# Session state
# =========================================================

if "analysis_ready" not in st.session_state:
    st.session_state.analysis_ready = False

if "handoff_generated" not in st.session_state:
    st.session_state.handoff_generated = False

if "escalation_reason" not in st.session_state:
    st.session_state.escalation_reason = ""


# =========================================================
# App header
# =========================================================

st.title("RideOps L1 Escalation Assistant")

st.write(
    "Use historical ticket patterns to make better pre-escalation decisions."
)


# =========================================================
# Ticket input
# =========================================================

st.subheader("Analyze a new ticket")

ticket_description = st.text_area(
    "Ticket description",
    placeholder=(
        "Example: Several employees are unable to see today's roster "
        "after a shift update."
    ),
    height=120,
)

categories = sorted(
    df["category"].dropna().unique()
)

selected_category = st.selectbox(
    "Category",
    categories,
)


# =========================================================
# L1 investigation
# =========================================================

st.subheader("L1 investigation")

l1_checks = st.text_area(
    "L1 checks completed",
    placeholder=(
        "Example: Confirmed the shift is active and the affected "
        "employees are assigned to the shift."
    ),
    height=100,
)

additional_context = st.text_area(
    "Additional context / evidence",
    placeholder=(
        "Example: Issue affects 12 employees across the same client "
        "and started after the 06:00 shift update."
    ),
    height=100,
)


# =========================================================
# Analyze button
# =========================================================

analyze_clicked = st.button(
    "Analyze Ticket",
    type="primary",
)


# =========================================================
# Ticket analysis
# =========================================================

if analyze_clicked:

    if not ticket_description.strip():

        st.warning("Please enter a ticket description.")

    else:

        # Reset workflow state
        st.session_state.analysis_ready = True
        st.session_state.handoff_generated = False
        st.session_state.escalation_reason = ""

        # Save current ticket information
        st.session_state.ticket_description = ticket_description
        st.session_state.selected_category = selected_category
        st.session_state.l1_checks = l1_checks
        st.session_state.additional_context = additional_context

        # -------------------------------------------------
        # Filter historical tickets by category
        # -------------------------------------------------

        category_df = df[
            df["category"] == selected_category
        ].copy()

        category_df = category_df[
            category_df["description"].notna()
        ].copy()

        # -------------------------------------------------
        # TF-IDF similarity
        # -------------------------------------------------

        documents = (
            category_df["description"]
            .astype(str)
            .tolist()
        )

        vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
        )

        historical_vectors = vectorizer.fit_transform(
            documents
        )

        new_ticket_vector = vectorizer.transform(
            [ticket_description]
        )

        similarity_scores = cosine_similarity(
            new_ticket_vector,
            historical_vectors,
        )[0]

        category_df["similarity"] = similarity_scores

        # -------------------------------------------------
        # Select relevant historical tickets
        # -------------------------------------------------

        relevant_tickets = category_df[
            category_df["similarity"] >= MIN_SIMILARITY
        ].copy()

        relevant_tickets = (
            relevant_tickets
            .sort_values("similarity", ascending=False)
            .head(MAX_SIMILAR_TICKETS)
            .copy()
        )

        evidence_count = len(relevant_tickets)

        # -------------------------------------------------
        # Evidence strength
        #
        # Evidence strength is based on the strongest
        # historical similarity rather than ticket count.
        # -------------------------------------------------

        if evidence_count == 0:

            evidence_strength = "Insufficient"

        else:

            top_similarity = relevant_tickets["similarity"].max()

            if top_similarity >= 0.60:
                evidence_strength = "Strong"

            elif top_similarity >= 0.40:
                evidence_strength = "Moderate"

            elif top_similarity >= 0.25:
                evidence_strength = "Weak"

            else:
                evidence_strength = "Insufficient"

        # -------------------------------------------------
        # Escalation rate among similar tickets
        # -------------------------------------------------

        escalation_rate = (
            relevant_tickets["escalated_to_eng"].eq("Yes").mean()
            if evidence_count > 0
            else 0
        )

        # -------------------------------------------------
        # Historical Engineering tickets
        # -------------------------------------------------

        escalated_matches = relevant_tickets[
            relevant_tickets["escalated_to_eng"] == "Yes"
        ].copy()

        escalated_count = len(escalated_matches)

        # -------------------------------------------------
        # Engineering outcomes
        # -------------------------------------------------

        if escalated_count > 0:

            outcome_distribution = (
                escalated_matches["eng_outcome"]
                .fillna("Unknown")
                .value_counts()
            )

        else:

            outcome_distribution = pd.Series(
                dtype="int64"
            )

        # -------------------------------------------------
        # Returned to L1
        # -------------------------------------------------

        returned_to_l1 = escalated_matches[
            escalated_matches["eng_outcome"].isin(
                [
                    "Insufficient info - returned to L1",
                    "Config issue - returned to L1",
                ]
            )
        ]

        returned_to_l1_rate = (
            len(returned_to_l1) / escalated_count
            if escalated_count > 0
            else 0
        )

        # -------------------------------------------------
        # Engineering handoffs
        # -------------------------------------------------

        median_handoffs = (
            escalated_matches["eng_handoffs"].median()
            if escalated_count > 0
            else None
        )

        # -------------------------------------------------
        # Resolution time
        # -------------------------------------------------

        relevant_tickets["created_at"] = pd.to_datetime(
            relevant_tickets["created_at"],
            errors="coerce",
        )

        relevant_tickets["final_closed_at"] = pd.to_datetime(
            relevant_tickets["final_closed_at"],
            errors="coerce",
        )

        relevant_tickets["resolution_hours"] = (
            relevant_tickets["final_closed_at"]
            - relevant_tickets["created_at"]
        ).dt.total_seconds() / 3600

        median_resolution = (
            relevant_tickets["resolution_hours"]
            .dropna()
            .median()
        )

        # -------------------------------------------------
        # Save analysis to session state
        # -------------------------------------------------

        st.session_state.relevant_tickets = relevant_tickets
        st.session_state.evidence_count = evidence_count
        st.session_state.evidence_strength = evidence_strength
        st.session_state.escalation_rate = escalation_rate
        st.session_state.escalated_count = escalated_count
        st.session_state.outcome_distribution = outcome_distribution
        st.session_state.returned_to_l1_rate = returned_to_l1_rate
        st.session_state.median_handoffs = median_handoffs
        st.session_state.median_resolution = median_resolution


# =========================================================
# Analysis results
# =========================================================

if st.session_state.analysis_ready:

    relevant_tickets = (
        st.session_state.relevant_tickets
    )

    evidence_count = (
        st.session_state.evidence_count
    )

    evidence_strength = (
        st.session_state.evidence_strength
    )

    escalation_rate = (
        st.session_state.escalation_rate
    )

    escalated_count = (
        st.session_state.escalated_count
    )

    outcome_distribution = (
        st.session_state.outcome_distribution
    )

    returned_to_l1_rate = (
        st.session_state.returned_to_l1_rate
    )

    median_handoffs = (
        st.session_state.median_handoffs
    )

    median_resolution = (
        st.session_state.median_resolution
    )

    # =====================================================
    # Historical evidence
    # =====================================================

    st.subheader("Historical evidence")

    col1, col2, col3 = st.columns(3)

    with col1:

        if evidence_count > 0:

            best_similarity = (
                relevant_tickets["similarity"].max()
            )

            st.metric(
                "Similar tickets",
                evidence_count,
                help=(
                    f"Best historical match: "
                    f"{best_similarity:.0%} similarity."
                ),
            )

        else:

            st.metric(
                "Similar tickets",
                0,
                help="No historical matches found.",
            )

    with col2:

        st.metric(
            "Escalation rate",
            f"{escalation_rate:.0%}",
        )

    with col3:

        if pd.notna(median_resolution):

            st.metric(
                "Median resolution",
                f"{median_resolution:.1f} hrs",
            )

        else:

            st.metric(
                "Median resolution",
                "N/A",
            )

    col4, col5, col6 = st.columns(3)

    with col4:

        if median_handoffs is not None:

            st.metric(
                "Median Engineering handoffs",
                f"{median_handoffs:.1f}",
            )

        else:

            st.metric(
                "Median Engineering handoffs",
                "N/A",
            )

    with col5:

        if escalated_count > 0:

            st.metric(
                "Returned to L1",
                f"{returned_to_l1_rate:.0%}",
            )

        else:

            st.metric(
                "Returned to L1",
                "N/A",
            )

    with col6:

        st.metric(
            "Evidence strength",
            evidence_strength,
        )


    # =====================================================
    # L1 guidance
    # =====================================================

    st.subheader("L1 guidance")

    if evidence_strength == "Insufficient":

        st.info(
            "There is not enough similar historical evidence to guide "
            "this ticket. Use normal L1 triage and document the "
            "investigation before deciding whether to escalate."
        )

    elif evidence_strength == "Weak":

        st.warning(
            "Historical matches are only weakly similar. Use normal "
            "L1 triage and do not rely on historical escalation "
            "patterns alone when deciding whether to escalate."
        )

    elif returned_to_l1_rate > 0:

        st.warning(
            f"{returned_to_l1_rate:.0%} of similar tickets that reached "
            "Engineering were returned to L1. Complete the relevant "
            "L1 checks and capture diagnostic context before escalating."
        )

    else:

        st.info(
            "No similar escalated tickets were returned to L1. "
            "If L1 checks do not resolve the issue, escalate with "
            "complete diagnostic context."
        )


    # =====================================================
    # Engineering outcomes
    # =====================================================

    st.subheader("Engineering outcomes")

    if not outcome_distribution.empty:

        outcome_df = (
            outcome_distribution
            .rename("tickets")
            .reset_index()
            .rename(
                columns={
                    "index": "outcome"
                }
            )
        )

        st.dataframe(
            outcome_df,
            use_container_width=True,
            hide_index=True,
        )

    else:

        st.info(
            "None of the similar historical tickets were escalated "
            "to Engineering."
        )


    # =====================================================
    # Engineering escalation
    # =====================================================

    st.subheader("Engineering escalation")

    st.write(
        "Escalate only if the L1 investigation indicates that "
        "Engineering involvement is required."
    )

    escalation_reason = st.text_area(
        "Why is Engineering needed?",
        value=st.session_state.escalation_reason,
        placeholder=(
            "Example: L1 checks are complete, but affected employees "
            "still cannot see the updated roster."
        ),
        height=100,
    )

    generate_handoff_clicked = st.button(
        "Generate Engineering Handoff",
        type="secondary",
    )

    if generate_handoff_clicked:

        if not escalation_reason.strip():

            st.warning(
                "Please explain why Engineering is needed before "
                "generating the handoff."
            )

        else:

            st.session_state.handoff_generated = True
            st.session_state.escalation_reason = (
                escalation_reason
            )


    # =====================================================
    # Engineering handoff
    # =====================================================

    if st.session_state.handoff_generated:

        st.success(
            "Engineering handoff generated."
        )

        saved_reason = (
            st.session_state.escalation_reason
        )

        saved_description = (
            st.session_state.ticket_description
        )

        saved_category = (
            st.session_state.selected_category
        )

        saved_l1_checks = (
            st.session_state.l1_checks
        )

        saved_context = (
            st.session_state.additional_context
        )

        escalation_context = f"""
Category:
{saved_category}

Issue:
{saved_description}

Why Engineering is needed:
{saved_reason}

L1 checks completed:
{saved_l1_checks if saved_l1_checks.strip() else "Not provided"}

Additional context / evidence:
{saved_context if saved_context.strip() else "Not provided"}

Historical evidence:
- Similar historical tickets: {evidence_count}
- Historical escalation rate: {escalation_rate:.0%}
- Historical median resolution: {
    f"{median_resolution:.1f} hours"
    if pd.notna(median_resolution)
    else "N/A"
}
- Historical median Engineering handoffs: {
    f"{median_handoffs:.1f}"
    if median_handoffs is not None
    else "N/A"
}
- Returned to L1 among escalated similar tickets: {
    f"{returned_to_l1_rate:.0%}"
    if escalated_count > 0
    else "N/A"
}

Historical Engineering outcomes:
"""

        if not outcome_distribution.empty:

            for outcome, count in (
                outcome_distribution.items()
            ):

                escalation_context += (
                    f"- {outcome}: {count}\n"
                )

        else:

            escalation_context += (
                "- No historical Engineering outcomes available.\n"
            )

        st.text_area(
            "Copy this context into the Engineering handoff",
            value=escalation_context.strip(),
            height=450,
        )


    # =====================================================
    # Most similar historical tickets
    # =====================================================

    st.subheader(
        "Most similar historical tickets"
    )

    display_columns = [
        "ticket_id",
        "description",
        "similarity",
        "escalated_to_eng",
        "eng_outcome",
        "eng_handoffs",
        "status",
    ]

    st.dataframe(
        relevant_tickets.head(5)[display_columns],
        use_container_width=True,
    )
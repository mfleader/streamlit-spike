import pandas as pd
import streamlit as st


import small_data
import ix_table


# hide_menu_style = """
#         <style>
#         #MainMenu {visibility: hidden;}
#         </style>
#         """
# st.markdown(hide_menu_style, unsafe_allow_html=True)

default_system_metrics = [
    "ninetyNinthEtcdDiskBackendCommitDurationSeconds_avg",
    "ninetyNinthEtcdDiskWalFsyncDurationSeconds_avg",
    "readOnlyAPICallsLatency_avg",
    "mutatingAPICallsLatency_avg",
    "podLatencyMeasurement_avg",
    "podLatencyQuantilesMeasurement_avg",
    "serviceSyncLatency_avg",
    "nodeCPU_Masters_avg",
    "podStatusCount_avg"
]


@st.experimental_memo
def filter_data(
    df: pd.DataFrame, account_selections: list[str], symbol_selections: list[str]
) -> pd.DataFrame:
    """
    Returns Dataframe with only accounts and symbols selected
    Args:
        df (pd.DataFrame): tidy data
        account_selections (list[str]): list of account names to include
        symbol_selections (list[str]): list of symbols to include
    Returns:
        pd.DataFrame: data only for the given accounts and symbols
    """
    df = df.copy()
    df = df[
        df.account_name.isin(account_selections) & df.symbol.isin(symbol_selections)
    ]

    return df



def main():
    data = small_data.data

    st.set_page_config(
        layout="centered", page_icon="🖱️", page_title="Interactive table app"
    )
    st.title("🖱️ Interactive table app")

    df = pd.DataFrame.from_records(data)

    has_job_ids = st.radio(
        'Do you know the workload job UUIDs?',
        ['yes', 'only one', 'no']
    )

    if has_job_ids == 'yes':
        col_a, col_b = st.columns(2)
        job_uuids = list(df['uuid'])

        with col_a:
            job_a = st.selectbox(
                'Group A Job UUID',
                options=job_uuids,
            )

        with col_b:
            job_b = st.selectbox(
                'Group B Job UUID',
                options=job_uuids,
            )
        ix_table.main(
            df[df['uuid'].isin((job_a, job_b))]
        )
    elif has_job_ids == 'only one':

        job_uuids = list(df['uuid'])

        job_a = st.selectbox(
            'Group A Job UUID',
            options=job_uuids,
        )

        platforms = list(df['platform'].unique())
        platform_selections = st.multiselect(
            'Cloud Platform',
            options=platforms,
            default=platforms
        )
        df = df[df['platform'].isin(platform_selections)]
        ix_table.main(
            df[df['uuid'] == job_a]
        )

    elif has_job_ids == 'no':
        st.write("Go ahead, click on a row in the table below!")
        ix_table.main(df)


    # with st.expander("Advanced Data Access"):
    #     st.write("Go ahead, click on a row in the table below!")
    #     ix_table.main(df)


if __name__ == '__main__':
    main()
import pandas as pd
import streamlit as st
import sqlmodel as sqm
from sqlmodel import select
from plotly.tools import mpl_to_plotly as ggplotly
import plotnine as p9

import config
from model import Run_Metrics
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


#  0fd17f48-node-density-20220601
#  889f9087-node-density-20220608
#  903e0658-node-density-20220506
#  099df96e-node-density-20220601
#  a91d1a8d-node-density-20220525
#  58e53ccf-node-density-heavy-20220601
#  a8a27db7-node-density-cni-20220601
#  5d6138f5-node-density-cni-20220525



def main():
    job_id = '0fd17f48-node-density-20220601'

    cfg = config.get_config()
    engine = sqm.create_engine(
        url = (
            f"{cfg.get('database.dialect')}://"
            f"{cfg.get('database.user')}:"
            f"{cfg.get('database.password')}@"
            f"{cfg.get('database.server_url')}:"
            f"{cfg.get('database.port')}/"
            f"{cfg.get('database.name')}"),
        echo = True
    )

    with sqm.Session(engine) as session:
        # run_metrics = session.exec(
        #     select(Run_Metrics).where(Run_Metrics.uuid == job_id)
        # )
        # print(run_metrics)
        job_uuids = session.exec(
            select(Run_Metrics.uuid)
        ).all()
        data = session.exec(
            select(Run_Metrics)
        ).all()


    df = pd.DataFrame.from_records(
        (d.dict() for d in data)
    )

    # print(df)

    # print(job_uuids)

    # print(df['uuid'])


    st.set_page_config(
        layout="centered", page_icon="🖱️", page_title="Interactive table app"
    )
    st.title("🖱️ Interactive table app")

    has_job_ids = st.radio(
        'Do you know the workload job UUIDs?',
        ['yes', 'only one', 'no']
    )

    # print(df)
    # print('======================')
    # print(df.isnull())
    # print('======================')
    # print(df.isna())
    # print('======================')



    if has_job_ids == 'yes':
        col_a, col_b = st.columns(2)
        # job_uuids = list(df['uuid'])

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


    st.markdown("Quantitative Statistics")
    st.sidebar.subheader("Graph Settings")
    chart_choice = st.sidebar.radio("",["QQplot","Boxplot","Histogram"])
    p = p9.ggplot(df)
    top = st.columns((1,1))
    bottom = st.columns(1)

    global numeric_columns
    global non_numeric_columns
    try:
        numeric_columns = list(df.select_dtypes(['float','int']).columns)
        non_numeric_columns = list(df.select_dtypes(['object']).columns)
        non_numeric_columns.append(None)
    except Exception as e:
        print(e)

    if chart_choice == "QQplot":
        with top[1]:
            x = st.selectbox('X-Axis', options=numeric_columns)
            cv = st.selectbox('Color', options=non_numeric_columns)
        if cv != None:
            p = (
                p +
                p9.stat_qq(p9.aes(sample=x, color=cv)) +
                p9.stat_qq_line(p9.aes(sample=x, color=cv)) +
                p9.labs(x = "Theoretical Quantiles", y = "Sample Quantiles")
            )
        else:
           p = (
                p +
                p9.stat_qq(p9.aes(sample=x)) +
                p9.stat_qq_line(p9.aes(sample=x)) +
                p9.labs(x = "Theoretical Quantiles", y = "Sample Quantiles")
            )

    if chart_choice == 'Boxplot':
        with top[1]:
            x = st.selectbox('X-Axis', options=numeric_columns)
            cv = st.selectbox("Color", options=non_numeric_columns)
        if cv != None:
            p = p + p9.geom_boxplot(p9.aes(x=cv,y=x, fill = cv)) + p9.coord_flip()
        else:
            p = p + p9.geom_boxplot(p9.aes(x=1,y=x,width=.1),color="darkblue", fill="lightblue") + p9.coord_flip()

    if chart_choice == 'Histogram':
        with top[1]:
            x = st.selectbox('X-Axis', options=numeric_columns)
            cv = st.selectbox("Color", options=non_numeric_columns)
            bins = st.slider("Number of Bins", min_value=1,max_value=40, value=7)
        if cv != None:
            p = p + p9.geom_histogram(p9.aes(x=x, fill = cv, color = cv),position= "identity",alpha=.4, bins = bins)
        else:
            p = p + p9.geom_histogram(p9.aes(x=x),color="darkblue", fill="lightblue", bins = bins)


    with top[1]:
        st.pyplot(p9.ggplot.draw(p))

    with top[0]:
        st.write(df[[cv, x]])

    with bottom[0]:
        st.write(df.describe().T)
        if cv != None:
            st.write(df.groupby([cv]).describe())


    # with st.expander("Advanced Data Access"):
    #     st.write("Go ahead, click on a row in the table below!")
    #     ix_table.main(df)


if __name__ == '__main__':
    main()
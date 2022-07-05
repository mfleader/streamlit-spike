import pandas as pd
import streamlit as st
import sqlmodel as sqm
from sqlmodel import select
import plotnine as p9

from scipy.stats import lognorm, norm, invgauss, invgamma, gamma
# import scipy.stats as scistats

import statistics as stats
from math import log, exp
from dataclasses import dataclass


from model import Run_Metrics
import ix_table


# hide_menu_style = """
#         <style>
#         #MainMenu {visibility: hidden;}
#         </style>
#         """
# st.markdown(hide_menu_style, unsafe_allow_html=True)



class PerformanceRange:
    great_lo: float = 0
    great_hi: float = 0
    poor_hi: float = 0
    bad_hi: float = 0

    def __init__(self, sr: pd.Series, great_hi: float = None):
        if great_hi:
            self.great_hi = great_hi
        else:
            self.great_hi = sr.quantile(q=.1)
        self.great_lo = sr.min()
        self.bad_hi = sr.max()
        self.poor_hi = self.great_hi + .5 * (self.bad_hi - self.great_hi)




# @st.experimental_singleton
def get_session(_engine):
    with sqm.Session(_engine) as session:
        yield session


@st.experimental_singleton
def get_engine():
    return sqm.create_engine(
        url = (
            f"{st.secrets['database']['dialect']}://"
            f"{st.secrets['database']['user']}:"
            f"{st.secrets['database']['password']}@"
            f"{st.secrets['database']['server_url']}:"
            f"{st.secrets['database']['port']}/"
            f"{st.secrets['database']['name']}"),
        # echo = True
    )


def histogram_w_highlights(df: pd.DataFrame, job_selection: str, bins, kpi: str, highlights: PerformanceRange ):

    # print(df[kpi])
    selected_job_result = df[df['uuid'] == job_selection][kpi]

    p = p9.ggplot(df)
    p = p + p9.geom_histogram(p9.aes(kpi), color="darkblue", fill="lightblue", bins = bins) +\
    p9.geom_vline(xintercept = selected_job_result) +\
    p9.annotate(
        geom='text',
        label='your selected job',
        x = selected_job_result,
        y = -300, ) +\
    p9.annotate(geom='rect',
        xmin = highlights.great_lo,
        xmax = highlights.great_hi,
        ymin = -50, ymax = 0,
        fill = 'green') +\
    p9.annotate(geom='text', label='great', color = 'green', x = highlights.great_lo, y = -800, )

    if highlights.poor_hi > highlights.great_hi:
        p = p + p9.annotate(
            geom='rect',
            xmin = highlights.great_hi,
            xmax = highlights.poor_hi,
            ymin = -50, ymax = 0, fill = 'yellow') +\
        p9.annotate(geom='text', label='slow disk', color = 'yellow',
            x = highlights.great_hi, y = -800, ) +\
        p9.annotate(
            geom='rect',
            xmin = highlights.poor_hi,
            xmax = highlights.bad_hi,
            ymin = -50, ymax = 0, fill = 'red')


    return p


def simulated_draws(sample: pd.Series, n_draws: int = 10_000):
    summary = sample.describe()
    return gamma.rvs(
        size = n_draws,
        a = summary.loc['count'],
        loc = summary.loc['mean'],
        scale = summary.loc['std']
    )


def model_data_world(df: pd.DataFrame, kpi: str):
    return pd.concat((
        df[['uuid', kpi]],
        pd.DataFrame.from_records(
            ({'uuid': 'sim', kpi: x} for x in simulated_draws(df[kpi])))
    ))



def main():

    st.set_page_config(
        layout="centered", page_icon="🖱️", page_title="OpenShift KPIs"
    )

    engine = get_engine()
    session = next(get_session(engine))
    job_uuids = session.exec(
        select(Run_Metrics.uuid)
    ).all()
    data = session.exec(
        select(Run_Metrics)
    ).all()



    df_og = pd.DataFrame.from_records(
        (d.dict() for d in data)
    )

    # job_uuid_select = st.columns(1)

    # with job_uuid_select:
    job_selection = st.selectbox(
        'Job UUID',
        options=job_uuids,
    )


    st.markdown("Pod Start Latency")

    df_og['pod_start_latency'] = df_og['podlatencyquantilesmeasurement_containersready_avg_p99'] -\
        df_og['podlatencyquantilesmeasurement_podscheduled_avg_p99']


    # print(log(psl_sumry.loc['mean']))
    # print(log(psl_sumry.loc['std']))
    # pod_latency_rng = norm.rvs(
    #     size = N,
    #     loc = log(psl_sumry.loc['mean']),
    #     scale = log(psl_sumry.loc['std'])
    # )
    # df_sim = pd.DataFrame.from_records(
    #     ({'uuid': 'sim', 'pod_start_latency': x} for x in pod_latency_rng)
    # )

    # pod_latency_rng = norm.rvs(
    #     size = N,
    #     loc = psl_sumry.loc['mean'],
    #     scale = psl_sumry.loc['std']
    # )
    # df_sim = pd.DataFrame.from_records(
    #     ({'uuid': 'sim', 'pod_start_latency': x} for x in pod_latency_rng)
    # )


    pod_start_ltcy_bins = st.slider("Number of Pod Start Latency Bins", min_value=1,max_value=40, value=22)
    pod_start_latency = model_data_world(df_og, 'pod_start_latency')
    pod_start_ltcy_grade_scale = PerformanceRange(
        pod_start_latency['pod_start_latency']
    )
    p1 = histogram_w_highlights(
        df=pod_start_latency,
        job_selection=job_selection,
        kpi='pod_start_latency',
        highlights=pod_start_ltcy_grade_scale,
        bins = pod_start_ltcy_bins,
    )
    st.pyplot(p9.ggplot.draw(p1))




    st.markdown('Etcd Health')


    with st.expander('Etcd Health Advanced'):
        etcd_write_dur_bins = st.slider("Number of Slow Disk Bins", min_value=1,max_value=40, value=22)
        etcd_write_dur = model_data_world(df_og, 'p99thetcddiskwalfsyncdurationseconds_avg')
        etcd_writes_dur_grade_scale = PerformanceRange(
            etcd_write_dur['p99thetcddiskwalfsyncdurationseconds_avg'],
            great_hi=.01)
        p2 = histogram_w_highlights(
            df=etcd_write_dur,
            job_selection=job_selection,
            kpi='p99thetcddiskwalfsyncdurationseconds_avg',
            highlights=etcd_writes_dur_grade_scale,
            bins = etcd_write_dur_bins
        )
        st.pyplot(p9.ggplot.draw(p2))


        etcd_leader_chg_rate_bins = st.slider("Number of Leader Rate Bins", min_value=1,max_value=40, value=22)
        etcd_leader_chg_rate = model_data_world(df_og, 'etcdleaderchangesrate_max')
        etcd_leader_chg_rate_grade_scale = PerformanceRange(
            etcd_leader_chg_rate['etcdleaderchangesrate_max']
        )
        p3 = histogram_w_highlights(
            df=etcd_leader_chg_rate,
            job_selection=job_selection,
            kpi='etcdleaderchangesrate_max',
            highlights=etcd_leader_chg_rate_grade_scale,
            bins = etcd_leader_chg_rate_bins
        )
        st.pyplot(p9.ggplot.draw(p3))

    # p99thetcdroundtriptimeseconds_avg: float
    # p99thetcddiskbackendcommitdurationseconds_avg: float




    # has_job_ids = st.radio(
    #     'Do you know the workload job UUIDs?',
    #     ['yes', 'no']
    # )

    # if has_job_ids == 'yes':
    # col_a, col_b = st.columns(2)

    # with col_a:
    #     job_a = st.selectbox(
    #         'Group A Job UUID',
    #         options=job_uuids,
    #     )

    # with col_b:
    #     job_b = st.multiselect(
    #         'Group B Job UUID',
    #         options=job_uuids,
    #     )

    # elif has_job_ids == 'only one':

    #     job_uuids = list(df['uuid'])

    #     job_a = st.selectbox(
    #         'Group A Job UUID',
    #         options=job_uuids,
    #     )

    #     platforms = list(df['platform'].unique())
    #     platform_selections = st.multiselect(
    #         'Cloud Platform',
    #         options=platforms,
    #         default=platforms
    #     )
    #     df = df[df['platform'].isin(platform_selections)]
    #     grid = ix_table.main(
    #         df[df['uuid'] == job_a]
    #     )
    # elif has_job_ids == 'no':
    #     st.write("Go ahead, click on a row in the table below!")
    #     grid = ix_table.main(df_og)
    #     df = grid['data']

    # if grid:
    #     st.write("You selected:")
    #     st.json(grid["selected_rows"])





    # st.markdown("Quantitative Statistics")
    # st.sidebar.subheader("Graph Settings")
    # chart_choice = st.sidebar.radio("",["QQplot","Boxplot","Histogram"])
    # # df = grid['data']
    # p = p9.ggplot(df)
    # top = st.columns((1,1))
    # bottom = st.columns(1)

    # global numeric_columns
    # global non_numeric_columns
    # try:
    #     numeric_columns = list(df.select_dtypes(['float','int']).columns)
    #     non_numeric_columns = list(df.select_dtypes(['object']).columns)
    #     non_numeric_columns.append(None)
    # except Exception as e:
    #     print(e)


    # if chart_choice == "QQplot":
    #     with top[1]:
    #         x = st.selectbox('X-Axis', options=numeric_columns)
    #         cv = st.selectbox('Color', options=non_numeric_columns)
    #     if cv != None:
    #         p = (
    #             p +
    #             p9.stat_qq(p9.aes(sample=x, color=cv)) +
    #             p9.stat_qq_line(p9.aes(sample=x, color=cv)) +
    #             p9.labs(x = "Theoretical Quantiles", y = "Sample Quantiles")
    #         )
    #     else:
    #        p = (
    #             p +
    #             p9.stat_qq(p9.aes(sample=x)) +
    #             p9.stat_qq_line(p9.aes(sample=x)) +
    #             p9.labs(x = "Theoretical Quantiles", y = "Sample Quantiles")
    #         )

    # if chart_choice == 'Boxplot':
    #     with top[1]:
    #         x = st.selectbox('X-Axis', options=numeric_columns)
    #         cv = st.selectbox("Color", options=non_numeric_columns)
    #     if cv != None:
    #         p = p + p9.geom_boxplot(p9.aes(x=cv,y=x, fill = cv)) + p9.coord_flip()
    #     else:
    #         p = p + p9.geom_boxplot(p9.aes(x=1,y=x,width=.1),color="darkblue", fill="lightblue") + p9.coord_flip()

    # if chart_choice == 'Histogram':
    #     with top[1]:
    #         x = st.selectbox('X-Axis', options=numeric_columns)
    #         cv = st.selectbox("Color", options=non_numeric_columns)
    #         bins = st.slider("Number of Bins", min_value=1,max_value=40, value=7)
    #     if cv != None:
    #         p = p + p9.geom_histogram(p9.aes(x=x, fill = cv, color = cv),position= "identity",alpha=.4, bins = bins)
    #     else:
    #         p = p + p9.geom_histogram(p9.aes(x=x),color="darkblue", fill="lightblue", bins = bins)


    # with top[1]:
    #     st.pyplot(p9.ggplot.draw(p))

    # with top[0]:
    #     st.write(df[[cv, x]])

    # with bottom[0]:
    #     st.write(df.describe().T)
    #     if cv != None:
    #         st.write(df.groupby([cv]).describe())


    # with st.expander("Advanced Data Access"):
    #     st.write("Go ahead, click on a row in the table below!")
    #     ix_table.main(df)


if __name__ == '__main__':
    main()
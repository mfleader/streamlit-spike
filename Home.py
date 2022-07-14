import pandas as pd
import numpy as np
import streamlit as st
import sqlmodel as sqm
from sqlalchemy.exc import OperationalError
from sqlmodel import select
from PerformanceRange import PerformanceRange
import gettext
import config
import plotly.express as px

from scipy.stats import lognorm, norm, invgauss, invgamma, gamma
# import scipy.stats as scistats

import statistics as stats
from math import log, exp

import numpy as np

# Locale Language File
lang_translations = gettext.translation('lang', localedir='locales/', languages=['en'])
lang_translations.install()
_ = lang_translations.gettext

# hide_menu_style = """
#         <style>
#         #MainMenu {visibility: hidden;}
#         </style>
#         """
# st.markdown(hide_menu_style, unsafe_allow_html=True)

# import cProfile, pstats

# import gevent
# profiler = cProfile.Profile()

# profiler.enable()


# @st.experimental_singleton
def get_session(_engine):
    with sqm.Session(_engine) as session:
        yield session


# @st.experimental_singleton
def get_engine():
    return sqm.create_engine(
        url = (
            f"{st.secrets['database']['dialect']}://"
            f"{st.secrets['database']['user']}:"
            f"{st.secrets['database']['password']}@"
            f"{st.secrets['database']['server_url']}:"
            f"{st.secrets['database']['port']}/"
            f"{st.secrets['database']['name']}"),
        connect_args = {
            "options": '-c statement_timeout=10'
        }
        # echo = True
    )


engine = get_engine()
from model import Run_Metrics


def histogram_w_highlights(df: pd.DataFrame, job_selection: str, bins, kpi: str, highlights: PerformanceRange, title=None ):

    # print(df[kpi])
    selected_job_result = df[df['uuid'] == job_selection][kpi][0]

    # Calculate histogram
    counts, bins = np.histogram(df[kpi], bins=60)
    bins = 0.5 * (bins[:-1] + bins[1:])

    # Calculate positions
    max_count = max(counts) + 10
    highlight_min = max_count * -0.05
    highlight_max = max_count * -0.28
    text_position = max_count * -0.18
    your_job_annotation_position = max_count * -0.3
    good_x = highlights.great_lo * 0.95 + highlights.great_hi * 0.05
    poor_x = highlights.great_hi * 0.95 + highlights.poor_hi * 0.05
    bad_x = highlights.poor_hi * 0.95 + highlights.bad_hi * 0.05
    # Create bar gragh
    fig = px.bar(x=bins, y=counts, height=250, labels={ "x": kpi, "y": "count"})
    # Add graphics to show performance
    fig.add_shape(type="rect", x0=highlights.great_lo, y0=highlight_min, x1=highlights.great_hi, y1=highlight_max, fillcolor="limegreen")
    fig.add_shape(type="rect", x0=highlights.great_hi, y0=highlight_min, x1=highlights.poor_hi, y1=highlight_max, fillcolor="yellow")
    fig.add_shape(type="rect", x0=highlights.poor_hi, y0=highlight_min, x1=highlights.bad_hi, y1=highlight_max, fillcolor="red")
    fig.add_annotation(text="Good", x = good_x, y=text_position, textangle=0, xanchor='left', showarrow=False,
        font=dict(size=15, color="black"))
    fig.add_annotation(text="Poor", x = poor_x, y=text_position, textangle=0, xanchor='left', showarrow=False,
        font=dict(size=15, color="black"))
    fig.add_annotation(text="Bad", x = bad_x, y=text_position, textangle=0, xanchor='left', showarrow=False,
        font=dict(size=15, color="white"))
    fig.update_yaxes(visible=True, showticklabels=False)

    # Indicate where the target job is
    fig.add_vline(x=selected_job_result, line_color = '#ff33bb')
    fig.add_annotation(text="Your Job Here", x = selected_job_result, y = 0, xanchor='left', ax=10,
        font=dict(size=15), arrowsize=2, arrowcolor="white", arrowhead=3)

    # Make the bar graph look like a histogram
    fig.update_layout(bargap = 0)
    fig.update_traces(
        marker_line_width=0
    )
    return fig

import math

def simulated_draws(sample: pd.Series, n_draws: int = 10_000):
    summary = sample.describe()
    return gamma.rvs(
        size = n_draws,
        a = summary.loc['count'],
        loc = .8 * (summary.loc['mean'] + .0000001),
        scale = summary.loc['std'] + summary.loc['mean'] / 10 + .00000001
        # scale = 10 * (summary.loc['std'] + 0.0000001)
    )




def model_data_world(df: pd.DataFrame, kpi: str):
    # simdf = simulated_draws(df[kpi])
    # print(simdf.describe())
    # print(np.mean(simdf))
    # print(np.median(simdf))
    # print(np.std(simdf))
    # print(np.max(simdf))
    return pd.concat((
        df[['uuid', kpi]],
        pd.DataFrame.from_records(
            ({'uuid': 'sim', kpi: x} for x in simulated_draws(df[kpi])))
    ))


def main():

    st.set_page_config(
        layout="centered", page_icon="🖱️", page_title="OpenShift KPIs"
    )

    st.title(_("DASHBOARD_TITLE"))

    data_col, cluster_col = st.columns(2)

    with data_col:
        st.subheader(_("DATA_SOURCES_TITLE"))
        selected_datasource = st.radio(_("SELECT_DATASOURCE"), ("PostgreSQL DB", "CSV"))

        if selected_datasource == "PostgreSQL DB":
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
        elif selected_datasource == "CSV":
            df_og = pd.read_csv(
                'data/run_metrics_2022-07-08.csv'
            )
        job_uuids = df_og['uuid'].values.tolist()

            # with job_uuid_select:
        job_selection = st.selectbox(
            _("SELECT_UUID"),
            options=job_uuids,
        )



    # job_uuid_select = st.columns(1)

    cluster_selection_df = df_og.loc[df_og['uuid'] == job_selection]

    similar_clusters = df_og.loc[
        (df_og['ocp_version'] == cluster_selection_df['ocp_version'].values[0]) &
        (df_og['platform'] == cluster_selection_df['platform'].values[0]) &
        (df_og['sdn_type'] == cluster_selection_df['sdn_type'].values[0]) &
        (df_og['workload'] == cluster_selection_df['workload'].values[0])
    ]

    # use unmatched cluster data when your matched cluster sample is too small
    if similar_clusters.shape[0] < 3:
        similar_clusters = df_og


    with cluster_col:
        st.subheader(_("CLUSTER_INFO_TITLE"))
        st.metric(
            label = _("OPENSHIFT_VERSION_METRIC"),
            value = cluster_selection_df['ocp_version'].values[0]
        )
        st.metric(
            label = _("PLATFORM_METRIC"),
            value = cluster_selection_df['platform'].values[0]
        )
        st.metric(
            label = _("CNI_METRIC"),
            value = cluster_selection_df['sdn_type'].values[0]
        )
        # st.metric(
        #     label = 'Data Collection Date',
        #     value = str(pd.Timestamp.fromtimestamp(int(df_og[df_og['uuid'] == job_selection]['timestamp'].values[0] / 1_000_000),'UTC'))
        # )




    st.header(_("DIFF_INSTANCE_Q_TITLE"))


    worker_cpu_col, control_cpu_col = st.columns(2)

    control_cpu = similar_clusters[similar_clusters['uuid'] == job_selection]['nodecpu_masters_avg'].values[0]
    control_cpu_agg = similar_clusters['nodecpu_masters_avg'].mean()
    worker_cpu = similar_clusters[similar_clusters['uuid'] == job_selection]['nodecpu_workers_avg'].values[0]
    worker_cpu_agg = similar_clusters['nodecpu_workers_avg'].mean()

    with control_cpu_col:
        st.metric(
            label = _("CONTROL_NODE_CPU"),
            value = round(control_cpu,2),
            delta = round(control_cpu - control_cpu_agg, 2),
            delta_color = 'inverse',
        )

    with worker_cpu_col:
        st.metric(
            label = _("WORKER_NODE_CPU"),
            value = round(worker_cpu,2),
            delta = round(worker_cpu - worker_cpu_agg,2),
            delta_color = 'inverse',
        )

    similar_clusters['pod_start_latency'] = similar_clusters['podlatencyquantilesmeasurement_containersready_avg_p99'] -\
        similar_clusters['podlatencyquantilesmeasurement_podscheduled_avg_p99']

    pod_latency = similar_clusters[similar_clusters['uuid'] == job_selection]['pod_start_latency'].values[0]
    pod_latency_agg = similar_clusters['pod_start_latency'].mean()

    # print(similar_clusters[[ 'workload', 'uuid', 'pod_start_latency']])

    st.metric(
        label = _("POD_LATENCY"),
        value = round(pod_latency, 2),
        delta = round(pod_latency - pod_latency_agg, 2),
        delta_color = 'inverse',
    )

    with st.expander(_("POD_LATENCY_ADVANCED")):

        pod_start_ltcy_bins = st.slider(_("POD_START_LATENCY_BINS"), min_value=1,max_value=40, value=22)
        pod_start_latency = model_data_world(similar_clusters, 'pod_start_latency')
        pod_start_ltcy_grade_scale = config.get_thresholds("", "", "pod_start_latency", pod_start_latency['pod_start_latency'])

        p1 = histogram_w_highlights(
            df=pod_start_latency,
            job_selection=job_selection,
            kpi='pod_start_latency',
            highlights=pod_start_ltcy_grade_scale,
            bins = pod_start_ltcy_bins,
            title = _("POD_START_LATENCY_TITLE")
        )
        st.plotly_chart(p1)
        #st.plotly_chart(p1)




    # st.markdown('Etcd Health')

    # df_og['etcd_health']
    etcdf = similar_clusters[[
    'uuid',
    'p99thetcdroundtriptimeseconds_avg',
    'p99thetcddiskbackendcommitdurationseconds_avg',
    'p99thetcddiskwalfsyncdurationseconds_avg',
    'etcdleaderchangesrate_max'
    ]]

    etcdf.loc[:,'health_score'] = etcdf[[ 'p99thetcdroundtriptimeseconds_avg',
    'p99thetcddiskbackendcommitdurationseconds_avg',
    'p99thetcddiskwalfsyncdurationseconds_avg',
    'etcdleaderchangesrate_max']].apply(lambda x: x <= x.quantile(q=.5)).astype(int).apply(pd.DataFrame.sum, axis=1)

    # etcdf['health_score_pct'] = etcdf['health_score'] / 4

    # p_etcd = (
    #     p9.ggplot(etcdf[etcdf['uuid'] == job_selection], p9.aes(x='uuid', y='health_score_pct')) +
    #     p9.geom_col(p9.aes(fill = 'health_score_pct')) +
    #     p9.scale_fill_gradient2(low = 'red', mid = 'yellow', high = 'green') +
    #     p9.ylim(0,1) +
    #     p9.coord_flip()
    # )

    # st.plotly_chart(p_etcd)

    etcd_health_score = etcdf[etcdf['uuid'] == job_selection]['health_score'].values[0]
    etcd_health_score_agg = etcdf['health_score'].mean()

    st.metric(
        label=_("ETCD_HEALTH_CHECKS_PASSED"),
        value = str(round(etcd_health_score, 2)) + ' (out of 4)',
        delta = round(etcd_health_score - etcd_health_score_agg, 2),
        delta_color = 'normal'
    )



    print(similar_clusters[[ 'workload', 'uuid', 'p99thetcddiskwalfsyncdurationseconds_avg']])

    print(similar_clusters[[ 'workload', 'uuid', 'etcdleaderchangesrate_max']])
    print('=============================================')


    with st.expander(_("ETCD_HEALTH_ADVANCED")):
        etcd_write_dur_bins = st.slider(_("SYNC_DURATION_BINS"), min_value=1,max_value=40, value=22)
        etcd_write_dur = model_data_world(similar_clusters, 'p99thetcddiskwalfsyncdurationseconds_avg')
        etcd_writes_dur_grade_scale = config.get_thresholds("", "", "etcd_disk_sync_duration",
            etcd_write_dur['p99thetcddiskwalfsyncdurationseconds_avg'])
        p2 = histogram_w_highlights(
            df=etcd_write_dur,
            job_selection=job_selection,
            kpi='p99thetcddiskwalfsyncdurationseconds_avg',
            highlights=etcd_writes_dur_grade_scale,
            bins = etcd_write_dur_bins,
            title = _("SYNC_DURATION_CHART_TITLE")
        )
        st.plotly_chart(p2)


        etcd_leader_chg_rate_bins = st.slider(_("LEADER_RATE_CHANGE_BINS"), min_value=1,max_value=40, value=22)
        etcd_leader_chg_rate = model_data_world(similar_clusters, 'etcdleaderchangesrate_max')
        etcd_leader_chg_rate_grade_scale = config.get_thresholds("", "", "etcd_leader_change_rate", etcd_leader_chg_rate['etcdleaderchangesrate_max'])
        p3 = histogram_w_highlights(
            df=etcd_leader_chg_rate,
            job_selection=job_selection,
            kpi='etcdleaderchangesrate_max',
            highlights=etcd_leader_chg_rate_grade_scale,
            bins = etcd_leader_chg_rate_bins,
            title = _("LEADER_CHANGE_RATE_CHART_TITLE")
        )
        st.plotly_chart(p3)






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
    #     st.plotly_chart(p)

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



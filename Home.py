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
            "options": '-c statement_timeout=20'
        }
        # echo = True
    )


engine = get_engine()
from model import Run_Metrics


def histogram_w_highlights(df: pd.DataFrame, job_selection: str, bins, kpi: str, highlights: PerformanceRange, title=None ):

    # print(df[kpi])
    selected_job_result = df[df['uuid'] == job_selection][kpi].values[0]

    # Calculate histogram
    counts, bins = np.histogram(df[kpi], bins=bins)
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
    fig = px.bar(x=bins, y=counts, height=220, labels={ "x": kpi, "y": "count"}, title=title)
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

    # Make the bar graph look like a histogram and fit properly
    fig.update_layout(bargap = 0, margin=dict(l=20, r=20, t=40, b=20))
    fig.update_traces(
        marker_line_width=0
    )
    return fig


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
    return pd.concat((
        df[['uuid', kpi]],
        pd.DataFrame.from_records(
            ({'uuid': 'sim', kpi: x} for x in simulated_draws(df[kpi])))
    ))


def main():

    st.set_page_config(
        layout="wide", page_icon="🖱️", page_title="OpenShift KPIs"
    )

    st.title(_("DASHBOARD_TITLE"))

    data_src_sidebar_container = st.sidebar.container()

    with data_src_sidebar_container:
        st.subheader(_("DATA_SOURCES_TITLE"))
        selected_datasource = st.radio(_("SELECT_DATASOURCE"), ("CSV", "PostgreSQL DB"))

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

        job_selection = st.selectbox(
            _("SELECT_UUID"),
            options=job_uuids,
        )

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

    cluster_side_container = st.sidebar.container()

    with cluster_side_container:
        st.header(_("CLUSTER_INFO_TITLE"))
        st.metric(
            label = _("OPENSHIFT_VERSION_METRIC"),
            value = cluster_selection_df['ocp_version'].values[0]
        )
        st.metric(
            label = _("CNI_METRIC"),
            value = cluster_selection_df['sdn_type'].values[0]
        )
        st.metric(
            label = _("PLATFORM_METRIC"),
            value = cluster_selection_df['platform'].values[0]
        )
        st.metric(
            label = _("CONTROL_NODES_TYPE_METRIC"),
            value = cluster_selection_df['master_nodes_type'].values[0]
        )


    # ========================================================= #
    # Different instance section
    # ========================================================= #
    st.header(_("DIFF_INSTANCE_Q_TITLE"))
    worker_nodes_col, control_nodes_col  = st.columns(2)

    ec2_instance_data = pd.DataFrame(
        data = [
            {
                'platform': 'ROSA',
                'instance_type': 'm5.4xlarge',
                'vcpu': 16,
                'memory': 64,
            },
            {
                'platform': 'ROSA',
                'instance_type': 'm5.2xlarge',
                'vcpu': 8,
                'memory': 32,
            },
            {
                'platform': 'AWS',
                'instance_type': 'm5.2xlarge',
                'vcpu': 8,
                'memory': 32,
            },
            {
                'platform': 'AWS',
                'instance_type': 'r5.4xlarge',
                'vcpu': 16,
                'memory': 128
            }
        ]
    )


    cluster = similar_clusters[similar_clusters['uuid'] == job_selection]
    control_cpu = cluster['nodecpu_masters_avg'].values[0]
    worker_cpu = cluster['nodecpu_workers_avg'].values[0]
    worker_mem = cluster['nodememoryutilization_workers_avg'].values[0] / 1073741824

    control_cpu_delta = control_cpu - .85 * ec2_instance_data.loc[
            ( ec2_instance_data['platform'] == cluster['platform'].values[0] ) &
            ( ec2_instance_data['instance_type'] == cluster['master_nodes_type'].values[0] )
        ]['vcpu'].values[0]

    worker_cpu_delta = worker_cpu - .85 * ec2_instance_data.loc[
            ( ec2_instance_data['platform'] == cluster['platform'].values[0] ) &
            ( ec2_instance_data['instance_type'] == cluster['worker_nodes_type'].values[0] )
        ]['vcpu'].values[0]

    worker_mem_delta = worker_mem - .85 * ec2_instance_data.loc[
            ( ec2_instance_data['platform'] == cluster['platform'].values[0] ) &
            ( ec2_instance_data['instance_type'] == cluster['worker_nodes_type'].values[0] )
        ]['memory'].values[0]

    with worker_nodes_col:
        st.metric(
            label = _("WORKER_NODE_CPU"),
            value = round(worker_cpu,2),
            delta = round(worker_cpu_delta,2),
            delta_color = 'inverse',
        )
        st.metric(
            label = _("WORKER_NODE_MEM"),
            value = str(round(worker_mem, 2)) + ' GiB',
            delta = round(worker_mem_delta, 2),
            delta_color = 'inverse'
        )

    with control_nodes_col:
        st.metric(
            label = _("CONTROL_NODE_CPU"),
            value = round(control_cpu,2),
            delta = round(control_cpu_delta, 2),
            delta_color = 'inverse',
        )



    # ========================================================= #
    # Pod latency section
    # ========================================================= #

    # Calculations
    similar_clusters['pod_start_latency'] = similar_clusters['podlatencyquantilesmeasurement_containersready_avg_p99'] -\
        similar_clusters['podlatencyquantilesmeasurement_podscheduled_avg_p99']

    pod_latency = float(similar_clusters[similar_clusters['uuid'] == job_selection]['pod_start_latency'].values[0])

    pod_start_latency = model_data_world(similar_clusters, 'pod_start_latency')
    pod_start_ltcy_grade_scale = config.get_thresholds("", "", "pod_start_latency", pod_start_latency['pod_start_latency'])

    # Display elements
    st.markdown("""---""")
    st.header(_("POD_LATENCY_Q_TITLE"))

    latency_col_1, latency_col_2 = st.columns(2)

    with latency_col_1:
        st.markdown("##### " + _("POD_LATENCY" + pod_start_ltcy_grade_scale.get_msg_suffix(float(pod_latency))))

    with latency_col_2:
        st.metric(
            label = _("POD_LATENCY"),
            value = str(round(pod_latency, 2)) + ' ms',
            delta = round(pod_latency - pod_start_ltcy_grade_scale.great_hi, 2),
            delta_color = 'inverse',
            help = _("POD_LATENCY_EXPLANATION")
        )


    p1 = histogram_w_highlights(
        df=pod_start_latency,
        job_selection=job_selection,
        kpi='pod_start_latency',
        highlights=pod_start_ltcy_grade_scale,
        bins = 60,
        title = _("POD_START_LATENCY_TITLE")
    )
    st.plotly_chart(p1)

    with st.expander(_("POD_LATENCY_ADVANCED")):

        st.markdown("TODO")


    # ========================================================= #
    # Etcd health section
    # ========================================================= #

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


    etcd_health_grade_scale = config.get_thresholds("", "", "etcd_health", etcdf['health_score'])

    etcd_health_score = float(etcdf[etcdf['uuid'] == job_selection]['health_score'].values[0])
    etcd_health_score_agg = etcdf['health_score'].mean()


    st.markdown("""---""")

    st.header(_("ETCD_HEALTH_Q_TITLE"))


    etcd_col_1, etcd_col_2 = st.columns(2)

    with etcd_col_1:
        st.markdown("##### " + _("ETCD_HEALTH" + etcd_health_grade_scale.get_msg_suffix(float(etcd_health_score))))

    with etcd_col_2:

        st.metric(
            label=_("ETCD_HEALTH_CHECKS_PASSED"),
            value = str(round(etcd_health_score, 2)) + ' (out of 4)',
            delta = round(float(etcd_health_grade_scale.great_lo) - etcd_health_score, 1),
            delta_color = 'normal'
        )



    with st.expander(_("ETCD_HEALTH_ADVANCED")):
        etcd_write_dur = model_data_world(similar_clusters, 'p99thetcddiskwalfsyncdurationseconds_avg')
        etcd_write_dur_value = float(etcd_write_dur[etcd_write_dur['uuid'] == job_selection]['p99thetcddiskwalfsyncdurationseconds_avg'].values[0])
        etcd_writes_dur_grade_scale = config.get_thresholds("", "", "etcd_disk_sync_duration",
            etcd_write_dur['p99thetcddiskwalfsyncdurationseconds_avg'])

        st.markdown("##### " + _("FSYNC_DURATION" + etcd_writes_dur_grade_scale.get_msg_suffix(etcd_write_dur_value)))


        p2 = histogram_w_highlights(
            df=etcd_write_dur,
            job_selection=job_selection,
            kpi='p99thetcddiskwalfsyncdurationseconds_avg',
            highlights=etcd_writes_dur_grade_scale,
            bins = 40,
            title = _("SYNC_DURATION_CHART_TITLE")
        )
        st.plotly_chart(p2)

        st.markdown("""---""")

        etcd_leader_chg_rate = model_data_world(similar_clusters, 'etcdleaderchangesrate_max')
        etcd_leader_chg_rate_value = float(etcd_leader_chg_rate[etcd_leader_chg_rate['uuid'] == job_selection]['etcdleaderchangesrate_max'].values[0])
        etcd_leader_chg_rate_grade_scale = config.get_thresholds("", "", "etcd_leader_change_rate", etcd_leader_chg_rate['etcdleaderchangesrate_max'])

        st.markdown("##### " + _("ETCD_LEADER_CHANGES" + etcd_leader_chg_rate_grade_scale.get_msg_suffix(etcd_leader_chg_rate_value)))

        p3 = histogram_w_highlights(
            df=etcd_leader_chg_rate,
            job_selection=job_selection,
            kpi='etcdleaderchangesrate_max',
            highlights=etcd_leader_chg_rate_grade_scale,
            bins = 20,
            title = _("LEADER_CHANGE_RATE_CHART_TITLE")
        )
        st.plotly_chart(p3)









if __name__ == '__main__':
    main()



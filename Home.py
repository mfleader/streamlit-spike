import pandas as pd
import numpy as np
import streamlit as st
import sqlmodel as sqm
from sqlalchemy.exc import OperationalError
from sqlmodel import select
from PerformanceRange import PerformanceRange
import instance_data
import gettext
import config
import plotly.express as px
import plotly.graph_objects as go

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
            "options": '-c statement_timeout=200'
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

def resource_gauge(available, current, peak, title, unit):
    fig = go.Figure(go.Indicator(
        number = {'suffix': unit},
        domain = {'x': [0, 1], 'y': [0, 1]},
        value = current,
        mode = "gauge+number+delta",
        title = {'text': title},
        delta = {'reference': available * 0.85, 'increasing': {'color': "Red"}, 'decreasing': {'color': "Green"}},
        gauge = {
             'axis': {'range': [None, available]},
             'steps' : [
                 {'range': [0, available * 0.75], 'color': "limegreen"},
                 {'range': [available * 0.75, available * 0.85], 'color': "yellow"},
                 {'range': [available * 0.85, available], 'color': "red"},
                 {'range': [0, available], 'color': "#262730", 'thickness': 0.6},
                 {'range': [current, current], 'thickness': 0.6, 'line': {'color': 'white', 'width': 1}}
              ],
             'bar': {'color': '#636efa', 'thickness': 0.35},
             'threshold': {'line': {'color': "#636efa", 'width': 3}, 'thickness': 0.1, 'value': peak}
        }))
    fig.update_layout(height=200, width=300, margin=dict(l=40, r=20, t=40, b=20))
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
        layout="centered", page_icon="🖱️", page_title="OpenShift KPIs"
    )

    datasource = config.get_datasource()

    if datasource == "postgresql":
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
    elif datasource == "csv":
        df_og = pd.read_csv(
            'data/run_metrics_2022-07-08.csv'
        )


    st.title(_("DASHBOARD_TITLE"))

    data_src_sidebar_container = st.sidebar.container()

    with data_src_sidebar_container:
        st.subheader(_("DATA_SOURCES_TITLE"))
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
        st.metric(
            label = _("WORKER_NODES_TYPE_METRIC"),
            value = cluster_selection_df['worker_nodes_type'].values[0]
        )


    # ========================================================= #
    # Different instance section
    # ========================================================= #
    st.header(_("DIFF_INSTANCE_Q_TITLE"))
    worker_nodes_col, control_nodes_col  = st.columns(2)

    ec2_instance_data = instance_data.get_instance_data()
    ec2_instance_data['memory'] = ec2_instance_data['memory'] / 1024

    cluster = similar_clusters[similar_clusters['uuid'] == job_selection]
    control_cpu = cluster['nodecpu_masters_avg'].values[0]
    worker_cpu = cluster['nodecpu_workers_avg'].values[0]
    worker_mem = cluster['nodememoryutilization_workers_avg'].values[0] / 1073741824
    control_cpu_max = cluster['nodecpu_masters_max'].values[0]
    worker_cpu_max = cluster['nodecpu_workers_max'].values[0]
    worker_mem_max = cluster['nodememoryutilization_workers_max'].values[0] / 1073741824
    platform = cluster['platform'].values[0].lower()
    if platform == "rosa":
        platform = "aws"

    master_node_cpu = ec2_instance_data.loc[
            ( ec2_instance_data['platform'] == platform ) &
            ( ec2_instance_data['instance_type'] == cluster['master_nodes_type'].values[0] )
        ]['vcpu'].values[0]
    control_cpu_delta = control_cpu - 0.85 * master_node_cpu
    worker_node_cpu = ec2_instance_data.loc[
            ( ec2_instance_data['platform'] == platform ) &
            ( ec2_instance_data['instance_type'] == cluster['worker_nodes_type'].values[0] )
        ]['vcpu'].values[0]
    worker_cpu_delta = worker_cpu - 0.85 * worker_node_cpu
    worker_node_mem = ec2_instance_data.loc[
            ( ec2_instance_data['platform'] == platform ) &
            ( ec2_instance_data['instance_type'] == cluster['worker_nodes_type'].values[0] )
        ]['memory'].values[0]
    worker_mem_delta = worker_mem - 0.85 * worker_node_mem

    if control_cpu_delta > 0 or worker_cpu_delta > 0 or worker_mem_delta > 0:
        st.markdown("##### " + _("INSTANCES_POOR"), unsafe_allow_html=True)
        suggested = st.expander(_("INSTANCE_SUGGESTIONS_TITLE"))

        with suggested:
            st.markdown(_("INSTANCE_SUGGESTIONS"))
            hide_table_row_index = '''
                <style>
                tbody th {display:none}
                .blank {display:none}
                </style>
                '''
            st.markdown(hide_table_row_index, unsafe_allow_html=True)
            if control_cpu_delta > 0:
                instance_type = cluster_selection_df['master_nodes_type'].values[0]
                st.subheader(_("CONTROL_INSTANCE_SUGGESTIONS"))
                st.table(
                    instance_data.get_larger_instances(ec2_instance_data, instance_type, True, False).drop(columns=["series"])
                )
            if worker_cpu_delta > 0 or worker_mem_delta > 0:
                instance_type = cluster_selection_df['worker_nodes_type'].values[0]
                st.subheader(_("WORKER_INSTANCE_SUGGESTIONS"))
                st.table(
                    instance_data.get_larger_instances(ec2_instance_data, instance_type,
                        worker_cpu_delta > 0, worker_mem_delta > 0).drop(columns=["series"])
                )
    else:
        st.markdown("##### " + _("INSTANCES_SUFFICIENT"), unsafe_allow_html=True)

    with worker_nodes_col:
        g1 = resource_gauge(worker_node_cpu, worker_cpu, worker_cpu_max, _("WORKER_NODE_CPU_GRAPH_TITLE"), " Cores")
        st.plotly_chart(g1)

        g2 = resource_gauge(worker_node_mem, round(worker_mem, 2), worker_mem_max, _("WORKER_NODE_MEM_GRAPH_TITLE"), " GiB")
        st.plotly_chart(g2)

    with control_nodes_col:
        g3 = resource_gauge(master_node_cpu, round(control_cpu, 2), control_cpu_max, _("CONTROL_NODE_CPU_GRAPH_TITLE"), " Cores")
        st.plotly_chart(g3)


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
        st.markdown("##### " + _("POD_LATENCY" + pod_start_ltcy_grade_scale.get_msg_suffix(float(pod_latency))), unsafe_allow_html=True)

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

    etcd_health_grade_scale = config.get_thresholds("", "", "etcd_health", None)

    # Etcd health data
    # Fsync duration
    etcd_write_dur_values = model_data_world(similar_clusters, 'p99thetcddiskwalfsyncdurationseconds_avg')
    etcd_write_dur_values["p99thetcddiskwalfsyncdurationseconds_avg"] = etcd_write_dur_values["p99thetcddiskwalfsyncdurationseconds_avg"].apply(lambda x: x * 1000) # Convert to ms
    etcd_write_dur_value = float(etcd_write_dur_values[etcd_write_dur_values['uuid'] == job_selection]['p99thetcddiskwalfsyncdurationseconds_avg'].values[0])
    etcd_writes_dur_grade_scale = config.get_thresholds("", "", "etcd_disk_sync_duration",
        etcd_write_dur_values['p99thetcddiskwalfsyncdurationseconds_avg'])

    # Leader changes
    etcd_leader_chg_rate_values = model_data_world(similar_clusters, 'etcdleaderchangesrate_max')
    etcd_leader_chg_rate_value = float(etcd_leader_chg_rate_values[etcd_leader_chg_rate_values['uuid'] == job_selection]['etcdleaderchangesrate_max'].values[0])
    etcd_leader_chg_rate_grade_scale = config.get_thresholds("", "", "etcd_leader_change_rate", etcd_leader_chg_rate_values['etcdleaderchangesrate_max'])

    # Round trip latency
    etcd_rtt_values = model_data_world(similar_clusters, 'p99thetcdroundtriptimeseconds_avg')
    etcd_rtt_values["p99thetcdroundtriptimeseconds_avg"] = etcd_rtt_values["p99thetcdroundtriptimeseconds_avg"].apply(lambda x: x * 1000) # Convert to ms
    etcd_rtt_value = float(etcd_rtt_values[etcd_leader_chg_rate_values['uuid'] == job_selection]['p99thetcdroundtriptimeseconds_avg'].values[0])
    etcd_rtt_grade_scale = config.get_thresholds("", "", "p99_etcd_rtt_avg", etcd_rtt_values['p99thetcdroundtriptimeseconds_avg'])

    # Commit duration
    etcd_commit_dur_values = model_data_world(similar_clusters, 'p99thetcddiskbackendcommitdurationseconds_avg')
    etcd_commit_dur_values["p99thetcddiskbackendcommitdurationseconds_avg"] = \
        etcd_commit_dur_values["p99thetcddiskbackendcommitdurationseconds_avg"].apply(lambda x: x * 1000) # Convert to ms
    etcd_commit_dur_value = float(etcd_commit_dur_values[etcd_commit_dur_values['uuid'] == job_selection]['p99thetcddiskbackendcommitdurationseconds_avg'].values[0])
    etcd_commit_dur_grade_scale = config.get_thresholds("", "", "p99_etcd_commit_dur_avg", etcd_commit_dur_values['p99thetcddiskbackendcommitdurationseconds_avg'])

    etcd_health_score = 0

    if etcd_write_dur_value < etcd_writes_dur_grade_scale.great_hi:
        etcd_health_score += 1
    if etcd_leader_chg_rate_value < etcd_leader_chg_rate_grade_scale.great_hi:
        etcd_health_score += 1
    if etcd_rtt_value < etcd_rtt_grade_scale.great_hi:
        etcd_health_score += 1
    if etcd_commit_dur_value < etcd_commit_dur_grade_scale.great_hi:
        etcd_health_score += 1


    st.markdown("""---""")

    st.header(_("ETCD_HEALTH_Q_TITLE"))


    etcd_col_1, etcd_col_2 = st.columns(2)

    with etcd_col_1:
        st.markdown("##### " + _("ETCD_HEALTH" + etcd_health_grade_scale.get_msg_suffix(float(etcd_health_score))), unsafe_allow_html=True)

    with etcd_col_2:

        st.metric(
            label=_("ETCD_HEALTH_CHECKS_PASSED"),
            value = str(round(etcd_health_score, 2)) + ' (out of 4)',
            delta = round(float(etcd_health_grade_scale.great_hi) - etcd_health_score, 1),
            delta_color = 'normal'
        )



    with st.expander(_("ETCD_HEALTH_ADVANCED")):

        fsync_col_1, fsync_col_2 = st.columns(2)

        with fsync_col_1:
            st.markdown("##### " + _("FSYNC_DURATION" + etcd_writes_dur_grade_scale.get_msg_suffix(etcd_write_dur_value)), unsafe_allow_html=True)

        with fsync_col_2:
            st.metric(
                label = _("SYNC_DURATION_CHART_TITLE"),
                value = str(round(etcd_write_dur_value, 1)) + ' ms',
                delta = round(etcd_write_dur_value - etcd_writes_dur_grade_scale.great_hi, 1),
                delta_color = 'inverse',
                help = _("POD_LATENCY_EXPLANATION")
            )


        p2 = histogram_w_highlights(
            df=etcd_write_dur_values,
            job_selection=job_selection,
            kpi='p99thetcddiskwalfsyncdurationseconds_avg',
            highlights=etcd_writes_dur_grade_scale,
            bins = 40,
            title = _("SYNC_DURATION_CHART_TITLE")
        )
        st.plotly_chart(p2)

        st.markdown("""---""")

        leader_changes_col_1, leader_changes_col_2 = st.columns(2)

        with leader_changes_col_1:
            st.markdown("##### " + _("ETCD_LEADER_CHANGES" + etcd_leader_chg_rate_grade_scale.get_msg_suffix(etcd_leader_chg_rate_value)), unsafe_allow_html=True)

        with leader_changes_col_2:
            st.metric(
                label = _("LEADER_CHANGE_RATE_CHART_TITLE"),
                value = str(etcd_leader_chg_rate_value),
                delta = round(etcd_leader_chg_rate_value - etcd_leader_chg_rate_grade_scale.great_hi, 4),
                delta_color = 'inverse',
                help = _("ETCD_LEADER_CHANGES_EXPLANATION")
            )

        p3 = histogram_w_highlights(
            df=etcd_leader_chg_rate_values,
            job_selection=job_selection,
            kpi='etcdleaderchangesrate_max',
            highlights=etcd_leader_chg_rate_grade_scale,
            bins = 20,
            title = _("LEADER_CHANGE_RATE_CHART_TITLE")
        )
        st.plotly_chart(p3)

        st.markdown("""---""")

        rtt_col_1, rtt_col_2 = st.columns(2)

        with rtt_col_1:
            st.markdown("##### " + _("ETCD_RTT" + etcd_rtt_grade_scale.get_msg_suffix(etcd_rtt_value)), unsafe_allow_html=True)

        with rtt_col_2:
            st.metric(
                label = _("ETCD_RTT_CHART_TITLE"),
                value = str(round(etcd_rtt_value, 1)) + ' ms',
                delta = round(etcd_rtt_value - etcd_rtt_grade_scale.great_hi, 1),
                delta_color = 'inverse',
                help = _("ETCD_RTT_EXPLANATION")
            )

        p4 = histogram_w_highlights(
            df=etcd_rtt_values,
            job_selection=job_selection,
            kpi='p99thetcdroundtriptimeseconds_avg',
            highlights=etcd_rtt_grade_scale,
            bins = 20,
            title = _("ETCD_RTT_CHART_TITLE")
        )
        st.plotly_chart(p4)

        st.markdown("""---""")

        commit_dur_col_1, commit_dur_col_2 = st.columns(2)

        with commit_dur_col_1:
            st.markdown("##### " + _("ETCD_COMMIT_DUR" + etcd_commit_dur_grade_scale.get_msg_suffix(etcd_commit_dur_value)), unsafe_allow_html=True)

        with commit_dur_col_2:
            st.metric(
                label = _("ETCD_COMMIT_DUR_CHART_TITLE"),
                value = str(round(etcd_commit_dur_value, 1)) + ' ms',
                delta = round(etcd_commit_dur_value - etcd_commit_dur_grade_scale.great_hi, 1),
                delta_color = 'inverse',
                help = _("ETCD_COMMIT_DUR_EXPLANATION")
            )

        p5 = histogram_w_highlights(
            df=etcd_commit_dur_values,
            job_selection=job_selection,
            kpi='p99thetcddiskbackendcommitdurationseconds_avg',
            highlights=etcd_commit_dur_grade_scale,
            bins = 20,
            title = _("ETCD_COMMIT_DUR_CHART_TITLE")
        )
        st.plotly_chart(p5)









if __name__ == '__main__':
    main()



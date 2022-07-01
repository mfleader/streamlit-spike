import pandas as pd
import streamlit as st
import sqlmodel as sqm
from sqlmodel import select
import plotnine as p9


from model import Run_Metrics
import ix_table


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



def main():

    engine = sqm.create_engine(
        url = (
            f"{st.secrets['database']['dialect']}://"
            f"{st.secrets['database']['user']}:"
            f"{st.secrets['database']['password']}@"
            f"{st.secrets['database']['server_url']}:"
            f"{st.secrets['database']['port']}/"
            f"{st.secrets['database']['name']}"),
        echo = True
    )
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


    st.set_page_config(
        layout="centered", page_icon="🖱️", page_title="Interactive table app"
    )
    st.title("🖱️ Interactive table app")

    has_job_ids = st.radio(
        'Do you know the workload job UUIDs?',
        ['yes', 'no']
    )

    if has_job_ids == 'yes':
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

        grid = ix_table.main(pd.DataFrame(df_og['uuid']))
        # print(job_uuids['data'])
        print(grid['selected_rows'])
        df = df_og[df_og['uuid'].isin(grid['selected_rows'])]
        print(df)

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
    elif has_job_ids == 'no':
        st.write("Go ahead, click on a row in the table below!")
        grid = ix_table.main(df_og)
        df = grid['data']

    if grid:
        st.write("You selected:")
        st.json(grid["selected_rows"])

    st.markdown("Quantitative Statistics")
    st.sidebar.subheader("Graph Settings")
    chart_choice = st.sidebar.radio("",["QQplot","Boxplot","Histogram"])
    # df = grid['data']
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
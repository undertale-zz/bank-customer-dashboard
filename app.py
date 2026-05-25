import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


MAIN_COLOR = "#2563eb"
BG_COLOR = "#f8fafc"
TEXT_COLOR = "#374151"


st.set_page_config(
    page_title="银行客户画像与流失风险分析系统",
    layout="wide"
)

st.title("银行客户画像与流失风险分析系统")

st.markdown(
    """
    <style>
    a[href^="#"] {
        display: none !important;
    }

    .block-container {
        padding-top: 4rem !important;
        padding-bottom: 2rem;
    }

    h1 {
        line-height: 1.25 !important;
        padding-top: 0.5rem !important;
        overflow: visible !important;
    }

    div[data-testid="stMetric"] {
        background-color: #f8fafc;
        padding: 14px 16px;
        border-radius: 12px;
        border: 1px solid #e5e7eb;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.info(
    "本系统用于分析银行客户特征，识别潜在流失客户，并根据客户风险等级和业务价值生成客户维护建议。"
)

uploaded_file = st.file_uploader("请上传银行客户 CSV 文件", type=["csv"])


def load_and_prepare_data(file):
    df = pd.read_csv(file)
    df.columns = df.columns.str.strip().str.lower()

    rename_map = {
        "country": "geography",
        "products_number": "numofproducts",
        "active_member": "isactivemember",
        "credit_card": "hascrcard",
        "estimated_salary": "estimatedsalary",
        "credit_score": "creditscore",
        "customer_id": "customerid"
    }

    df = df.rename(columns=rename_map)

    if "churn" in df.columns and "exited" not in df.columns:
        df["exited"] = df["churn"]

    return df


def generate_customer_segment(row, value_threshold):
    risk = row.get("churn_risk_probability", 0)
    balance = row.get("balance", 0)

    if risk >= 0.7 and balance >= value_threshold:
        return "高风险高价值客户"
    elif risk >= 0.7:
        return "高风险普通客户"
    elif risk < 0.7 and balance >= value_threshold:
        return "低风险高价值客户"
    else:
        return "稳定普通客户"


def generate_action(row):
    risk = row.get("churn_risk_probability", 0)
    balance = row.get("balance", 0)
    active = row.get("isactivemember", 1)
    products = row.get("numofproducts", 0)
    age = row.get("age", 0)

    if risk >= 0.7 and balance >= 100000:
        return "安排 VIP 客户经理回访，提供专属服务或手续费优惠"
    elif risk >= 0.7 and active == 0:
        return "发送客户激活活动，提高客户活跃度"
    elif risk >= 0.7 and products <= 1:
        return "推荐合适的银行产品，提高客户粘性"
    elif risk >= 0.7 and age >= 60:
        return "提供电话客服或线下服务支持"
    elif risk >= 0.7:
        return "进行客户关怀回访，了解潜在流失原因"
    else:
        return "保持常规客户维护"


def segment_label_for_chart(segment):
    segment_map = {
        "高风险高价值客户": "High Risk High Value",
        "高风险普通客户": "High Risk Normal",
        "低风险高价值客户": "Low Risk High Value",
        "稳定普通客户": "Stable Normal"
    }
    return segment_map.get(segment, str(segment))


def apply_plotly_layout(fig, height=320):
    fig.update_layout(
        height=height,
        margin=dict(l=20, r=20, t=50, b=30),
        plot_bgcolor=BG_COLOR,
        paper_bgcolor="#ffffff",
        title=dict(font=dict(size=14, color="#111827")),
        font=dict(size=12, color=TEXT_COLOR),
        hoverlabel=dict(bgcolor="white", font_size=12),
    )
    return fig


def show_bar_chart(df, x_col, y_col, title, x_label, y_label, is_percent=False, height=320):
    fig = px.bar(
        df,
        x=x_col,
        y=y_col,
        text=y_col,
        title=title
    )

    if is_percent:
        fig.update_traces(
            marker_color=MAIN_COLOR,
            texttemplate="%{y:.2%}",
            textposition="outside",
            hovertemplate=f"{x_label}: %{{x}}<br>{y_label}: %{{y:.2%}}<extra></extra>"
        )
        fig.update_layout(yaxis_tickformat=".0%")
    else:
        fig.update_traces(
            marker_color=MAIN_COLOR,
            texttemplate="%{y}",
            textposition="outside",
            hovertemplate=f"{x_label}: %{{x}}<br>{y_label}: %{{y}}<extra></extra>"
        )

    fig.update_layout(
        xaxis_title=x_label,
        yaxis_title=y_label
    )

    apply_plotly_layout(fig, height)
    st.plotly_chart(fig, use_container_width=True)


def show_histogram(df, column, title, x_label, y_label, bins=20, height=320):
    fig = px.histogram(
        df,
        x=column,
        nbins=bins,
        title=title
    )

    fig.update_traces(
        marker_color=MAIN_COLOR,
        marker_line_color="white",
        marker_line_width=0.8,
        hovertemplate=f"{x_label}: %{{x}}<br>{y_label}: %{{y}}<extra></extra>"
    )

    fig.update_layout(
        xaxis_title=x_label,
        yaxis_title=y_label
    )

    apply_plotly_layout(fig, height)
    st.plotly_chart(fig, use_container_width=True)


def show_horizontal_bar_chart(df, x_col, y_col, title, x_label, y_label, height=320):
    fig = px.bar(
        df,
        x=x_col,
        y=y_col,
        orientation="h",
        text=x_col,
        title=title
    )

    fig.update_traces(
        marker_color=MAIN_COLOR,
        texttemplate="%{x:.3f}",
        textposition="outside",
        hovertemplate=f"{y_label}: %{{y}}<br>{x_label}: %{{x:.4f}}<extra></extra>"
    )

    fig.update_layout(
        xaxis_title=x_label,
        yaxis_title=y_label,
        yaxis=dict(autorange="reversed")
    )

    apply_plotly_layout(fig, height)
    st.plotly_chart(fig, use_container_width=True)


def show_confusion_matrix(cm):
    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=["Predicted 0", "Predicted 1"],
            y=["True 0", "True 1"],
            colorscale="Blues",
            text=cm,
            texttemplate="%{text}",
            hovertemplate="True Label: %{y}<br>Predicted Label: %{x}<br>Count: %{z}<extra></extra>"
        )
    )

    fig.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted Label",
        yaxis_title="True Label"
    )

    apply_plotly_layout(fig, height=320)
    st.plotly_chart(fig, use_container_width=True)


if uploaded_file is not None:
    df = load_and_prepare_data(uploaded_file)

    st.sidebar.title("筛选条件")
    st.sidebar.write("可通过以下条件查看不同客户群体的特征和流失风险。")

    filtered_df = df.copy()

    if "geography" in df.columns:
        selected_geo = st.sidebar.multiselect(
            "选择地区",
            options=sorted(df["geography"].dropna().unique()),
            default=sorted(df["geography"].dropna().unique())
        )
        filtered_df = filtered_df[filtered_df["geography"].isin(selected_geo)]

    if "gender" in df.columns:
        selected_gender = st.sidebar.multiselect(
            "选择性别",
            options=sorted(df["gender"].dropna().unique()),
            default=sorted(df["gender"].dropna().unique())
        )
        filtered_df = filtered_df[filtered_df["gender"].isin(selected_gender)]

    if "age" in df.columns:
        min_age = int(df["age"].min())
        max_age = int(df["age"].max())
        selected_age = st.sidebar.slider(
            "选择年龄范围",
            min_value=min_age,
            max_value=max_age,
            value=(min_age, max_age)
        )
        filtered_df = filtered_df[
            (filtered_df["age"] >= selected_age[0]) &
            (filtered_df["age"] <= selected_age[1])
        ]

    if "isactivemember" in df.columns:
        selected_active = st.sidebar.multiselect(
            "是否活跃客户",
            options=sorted(df["isactivemember"].dropna().unique()),
            default=sorted(df["isactivemember"].dropna().unique())
        )
        filtered_df = filtered_df[filtered_df["isactivemember"].isin(selected_active)]

    if "numofproducts" in df.columns:
        selected_products = st.sidebar.multiselect(
            "产品持有数量",
            options=sorted(df["numofproducts"].dropna().unique()),
            default=sorted(df["numofproducts"].dropna().unique())
        )
        filtered_df = filtered_df[filtered_df["numofproducts"].isin(selected_products)]

    st.header("1. 项目概览")

    total_customers = len(filtered_df)

    if "exited" in filtered_df.columns and total_customers > 0:
        churn_count = filtered_df["exited"].sum()
        churn_rate = churn_count / total_customers
    else:
        churn_count = 0
        churn_rate = 0

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("筛选后客户数", total_customers)
    col2.metric("流失客户数", int(churn_count))
    col3.metric("流失率", f"{churn_rate:.2%}")

    if "balance" in filtered_df.columns and total_customers > 0:
        col4.metric("平均账户余额", f"{filtered_df['balance'].mean():,.2f}")
    else:
        col4.metric("平均账户余额", "N/A")

    st.subheader("数据预览")
    st.dataframe(filtered_df.head(), use_container_width=True)

    st.subheader("缺失值检查")
    missing_df = filtered_df.isnull().sum().reset_index()
    missing_df.columns = ["字段", "缺失值数量"]
    st.dataframe(missing_df, use_container_width=True)

    st.divider()

    st.header("2. 客户画像分析")

    col1, col2 = st.columns(2)

    with col1:
        if "age" in filtered_df.columns:
            st.subheader("年龄分布")
            show_histogram(
                filtered_df,
                "age",
                "Age Distribution",
                "Age",
                "Customers"
            )

    with col2:
        if "geography" in filtered_df.columns:
            st.subheader("地区客户数量图")
            geo_count = filtered_df["geography"].value_counts()
            geo_df = geo_count.reset_index()
            geo_df.columns = ["Region", "Customers"]

            show_bar_chart(
                geo_df,
                "Region",
                "Customers",
                "Customer Count by Region",
                "Region",
                "Customers"
            )

    col3, col4 = st.columns(2)

    with col3:
        if "gender" in filtered_df.columns:
            st.subheader("性别客户数量")
            gender_count = filtered_df["gender"].value_counts()
            gender_df = gender_count.reset_index()
            gender_df.columns = ["Gender", "Customers"]

            show_bar_chart(
                gender_df,
                "Gender",
                "Customers",
                "Customer Count by Gender",
                "Gender",
                "Customers"
            )

    with col4:
        if "numofproducts" in filtered_df.columns:
            st.subheader("产品持有数量图")
            product_count = filtered_df["numofproducts"].value_counts().sort_index()
            product_df = product_count.reset_index()
            product_df.columns = ["Number of Products", "Customers"]

            show_bar_chart(
                product_df,
                "Number of Products",
                "Customers",
                "Product Count Distribution",
                "Number of Products",
                "Customers"
            )

    if "balance" in filtered_df.columns:
        st.subheader("客户账户余额分布")
        show_histogram(
            filtered_df,
            "balance",
            "Balance Distribution",
            "Balance",
            "Customers"
        )

    st.divider()

    st.header("3. 流失风险分析")

    if "exited" in filtered_df.columns and len(filtered_df) > 0:
        col1, col2 = st.columns(2)

        with col1:
            if "age" in filtered_df.columns:
                st.subheader("不同年龄段流失率图")

                filtered_df = filtered_df.copy()
                filtered_df["age_group"] = pd.cut(
                    filtered_df["age"],
                    bins=[0, 30, 40, 50, 60, 100],
                    labels=["Under 30", "31-40", "41-50", "51-60", "Over 60"]
                )

                age_churn = filtered_df.groupby("age_group", observed=False)["exited"].mean()
                age_churn_df = age_churn.reset_index()
                age_churn_df.columns = ["Age Group", "Churn Rate"]

                show_bar_chart(
                    age_churn_df,
                    "Age Group",
                    "Churn Rate",
                    "Churn Rate by Age Group",
                    "Age Group",
                    "Churn Rate",
                    is_percent=True
                )

        with col2:
            if "geography" in filtered_df.columns:
                st.subheader("不同地区流失率图")

                geo_churn = filtered_df.groupby("geography")["exited"].mean()
                geo_churn_df = geo_churn.reset_index()
                geo_churn_df.columns = ["Region", "Churn Rate"]

                show_bar_chart(
                    geo_churn_df,
                    "Region",
                    "Churn Rate",
                    "Churn Rate by Region",
                    "Region",
                    "Churn Rate",
                    is_percent=True
                )

        col3, col4 = st.columns(2)

        with col3:
            if "numofproducts" in filtered_df.columns:
                st.subheader("不同产品持有数量流失率")

                product_churn = filtered_df.groupby("numofproducts")["exited"].mean()
                product_churn_df = product_churn.reset_index()
                product_churn_df.columns = ["Number of Products", "Churn Rate"]

                show_bar_chart(
                    product_churn_df,
                    "Number of Products",
                    "Churn Rate",
                    "Churn Rate by Product Count",
                    "Number of Products",
                    "Churn Rate",
                    is_percent=True
                )

        with col4:
            if "isactivemember" in filtered_df.columns:
                st.subheader("不同活跃状态流失率")

                active_churn = filtered_df.groupby("isactivemember")["exited"].mean()
                active_churn_df = active_churn.reset_index()
                active_churn_df.columns = ["Active Member", "Churn Rate"]
                active_churn_df["Active Member"] = active_churn_df["Active Member"].map({
                    0: "Inactive",
                    1: "Active"
                })

                show_bar_chart(
                    active_churn_df,
                    "Active Member",
                    "Churn Rate",
                    "Churn Rate by Active Status",
                    "Active Member",
                    "Churn Rate",
                    is_percent=True
                )

    else:
        st.warning("当前数据中没有找到流失标签字段，无法进行流失风险分析。")

    st.divider()

    st.header("4. 预测模型评估")

    if "exited" in df.columns:
        model_df = df.copy()

        drop_cols = [
            "rownumber",
            "customerid",
            "surname",
            "churn",
            "age_group",
            "customer_segment"
        ]

        model_df = model_df.drop(
            columns=[col for col in drop_cols if col in model_df.columns],
            errors="ignore"
        )

        for col in model_df.select_dtypes(include=["object", "category"]).columns:
            le = LabelEncoder()
            model_df[col] = le.fit_transform(model_df[col].astype(str))

        model_df = model_df.dropna()

        X = model_df.drop(columns=["exited"])
        y = model_df["exited"]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight="balanced"
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("准确率 Accuracy", f"{accuracy_score(y_test, y_pred):.3f}")
        col2.metric("召回率 Recall", f"{recall_score(y_test, y_pred):.3f}")
        col3.metric("F1 值", f"{f1_score(y_test, y_pred):.3f}")
        col4.metric("AUC 值", f"{roc_auc_score(y_test, y_prob):.3f}")

        st.caption("召回率在客户流失预测中较为重要，因为银行希望尽可能识别出真正可能流失的客户。")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("混淆矩阵")
            cm = confusion_matrix(y_test, y_pred)
            show_confusion_matrix(cm)

        with col2:
            st.subheader("特征重要性")
            importance = pd.DataFrame({
                "Feature": X.columns,
                "Importance": model.feature_importances_
            }).sort_values(by="Importance", ascending=False)

            top_importance = importance.head(8)

            show_horizontal_bar_chart(
                top_importance,
                "Importance",
                "Feature",
                "Top Feature Importance",
                "Importance",
                "Feature"
            )

        st.dataframe(importance, use_container_width=True)

        result_df = df.copy()

        predict_df = model_df.drop(columns=["exited"])
        result_df = result_df.loc[model_df.index]
        result_df["churn_risk_probability"] = model.predict_proba(predict_df)[:, 1]

        if "balance" in result_df.columns:
            value_threshold = result_df["balance"].quantile(0.75)
        else:
            value_threshold = 0

        result_df["customer_segment"] = result_df.apply(
            lambda row: generate_customer_segment(row, value_threshold),
            axis=1
        )

        result_df["suggested_action"] = result_df.apply(generate_action, axis=1)

        dashboard_df = result_df[result_df.index.isin(filtered_df.index)]

        if len(dashboard_df) == 0:
            dashboard_df = result_df.copy()

        high_risk_default_count = (dashboard_df["churn_risk_probability"] >= 0.7).sum()

        st.subheader("预测后客户概览")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("总客户数", len(dashboard_df))
        col2.metric("高风险客户数", int(high_risk_default_count))
        col3.metric("平均流失风险", f"{dashboard_df['churn_risk_probability'].mean():.2%}")

        if "balance" in result_df.columns:
            col4.metric("高价值客户阈值", f"{value_threshold:,.2f}")
        else:
            col4.metric("高价值客户阈值", "N/A")

        st.divider()

        st.header("5. 高风险客户名单")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("流失风险概率分布")
            show_histogram(
                dashboard_df,
                "churn_risk_probability",
                "Churn Risk Distribution",
                "Churn Risk Probability",
                "Customers"
            )

        with col2:
            st.subheader("Top 10 高风险客户图")

            top10 = dashboard_df.sort_values(
                by="churn_risk_probability",
                ascending=False
            ).head(10)

            if "customerid" in top10.columns:
                x_values = top10["customerid"].astype(str)
            else:
                x_values = top10.index.astype(str)

            top10_df = pd.DataFrame({
                "Customer ID": x_values,
                "Risk Probability": top10["churn_risk_probability"].values
            })

            show_bar_chart(
                top10_df,
                "Customer ID",
                "Risk Probability",
                "Top 10 High-Risk Customers",
                "Customer ID",
                "Risk Probability",
                is_percent=True
            )

        st.subheader("高风险客户筛选与导出")

        col1, col2 = st.columns(2)

        with col1:
            risk_threshold = st.slider(
                "选择高风险阈值",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                step=0.05
            )

        with col2:
            top_n = st.selectbox(
                "选择导出客户数量",
                options=[10, 20, 50, 100, 200],
                index=2
            )

        high_risk = dashboard_df[
            dashboard_df["churn_risk_probability"] >= risk_threshold
        ].sort_values(
            by="churn_risk_probability",
            ascending=False
        ).head(top_n)

        st.write(f"当前筛选出 {len(high_risk)} 名高风险客户。")

        display_cols = [
            "customerid",
            "geography",
            "gender",
            "age",
            "creditscore",
            "balance",
            "numofproducts",
            "isactivemember",
            "churn_risk_probability",
            "customer_segment",
            "suggested_action"
        ]

        display_cols = [col for col in display_cols if col in high_risk.columns]

        st.dataframe(high_risk[display_cols], use_container_width=True)

        csv = high_risk.to_csv(index=False).encode("utf-8-sig")

        st.download_button(
            label="下载高风险客户名单 CSV",
            data=csv,
            file_name="high_risk_customers.csv",
            mime="text/csv"
        )

        st.subheader("客户分层分析")

        segment_count = dashboard_df["customer_segment"].value_counts()
        segment_table = segment_count.reset_index()
        segment_table.columns = ["客户类型", "客户数量"]

        col1, col2 = st.columns(2)

        with col1:
            st.dataframe(segment_table, use_container_width=True)

        with col2:
            chart_segment_count = segment_count.copy()
            chart_segment_count.index = [
                segment_label_for_chart(segment)
                for segment in chart_segment_count.index
            ]

            segment_df = chart_segment_count.reset_index()
            segment_df.columns = ["Segment", "Customers"]

            show_bar_chart(
                segment_df,
                "Segment",
                "Customers",
                "Customer Segmentation",
                "Segment",
                "Customers"
            )

        st.divider()

        st.header("6. 业务建议")

        st.write("系统根据客户流失风险、账户余额、活跃状态、产品持有数量和年龄生成客户维护建议。")

        st.write("1. 对高风险高价值客户，优先安排客户经理回访。")
        st.write("2. 对高风险但不活跃客户，建议发送激活活动或优惠信息。")
        st.write("3. 对仅持有少量产品的高风险客户，建议进行交叉销售。")
        st.write("4. 对年龄较大的高风险客户，建议提供电话客服或线下服务支持。")
        st.write("5. 对低风险高价值客户，建议继续保持长期关系维护。")

    else:
        st.warning("当前数据中没有找到流失标签字段，无法训练预测模型。")

else:
    st.info("请先上传 CSV 文件。")

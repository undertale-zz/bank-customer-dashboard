import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)


st.set_page_config(
    page_title="银行客户流失风险分析系统",
    layout="wide"
)

st.title("银行客户流失风险分析系统")
st.write("本系统基于银行客户数据，支持客户画像分析、流失风险预测、客户分层、高风险客户导出与业务建议生成。")

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
        return "安排VIP客户经理回访，提供专属服务或手续费优惠"
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


if uploaded_file is not None:
    df = load_and_prepare_data(uploaded_file)

    st.sidebar.header("筛选条件")

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

    st.header("1. Overview")

    st.subheader("数据预览")
    st.dataframe(filtered_df.head())

    st.subheader("数据基本信息")

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
        col4.metric("平均余额", f"{filtered_df['balance'].mean():,.2f}")
    else:
        col4.metric("平均余额", "N/A")

    st.subheader("缺失值检查")
    missing_df = filtered_df.isnull().sum().reset_index()
    missing_df.columns = ["字段", "缺失值数量"]
    st.dataframe(missing_df)

    st.header("2. Customer Profile")

    col1, col2 = st.columns(2)

    with col1:
        if "age" in filtered_df.columns:
            fig, ax = plt.subplots()
            ax.hist(filtered_df["age"].dropna(), bins=20)
            ax.set_xlabel("Age")
            ax.set_ylabel("Number of Customers")
            ax.set_title("Age Distribution")
            st.pyplot(fig)

    with col2:
        if "geography" in filtered_df.columns:
            geo_count = filtered_df["geography"].value_counts()
            fig, ax = plt.subplots()
            ax.bar(geo_count.index.astype(str), geo_count.values)
            ax.set_xlabel("Region")
            ax.set_ylabel("Number of Customers")
            ax.set_title("Customer Distribution by Region")
            st.pyplot(fig)

    col3, col4 = st.columns(2)

    with col3:
        if "gender" in filtered_df.columns:
            gender_count = filtered_df["gender"].value_counts()
            fig, ax = plt.subplots()
            ax.bar(gender_count.index.astype(str), gender_count.values)
            ax.set_xlabel("Gender")
            ax.set_ylabel("Number of Customers")
            ax.set_title("Customer Distribution by Gender")
            st.pyplot(fig)

    with col4:
        if "numofproducts" in filtered_df.columns:
            product_count = filtered_df["numofproducts"].value_counts().sort_index()
            fig, ax = plt.subplots()
            ax.bar(product_count.index.astype(str), product_count.values)
            ax.set_xlabel("Number of Products")
            ax.set_ylabel("Number of Customers")
            ax.set_title("Customer Distribution by Number of Products")
            st.pyplot(fig)

    if "balance" in filtered_df.columns:
        st.subheader("客户余额分布")
        fig, ax = plt.subplots()
        ax.hist(filtered_df["balance"].dropna(), bins=20)
        ax.set_xlabel("Balance")
        ax.set_ylabel("Number of Customers")
        ax.set_title("Balance Distribution")
        st.pyplot(fig)

    st.header("3. Churn Analysis")

    if "exited" in filtered_df.columns and len(filtered_df) > 0:
        col1, col2 = st.columns(2)

        with col1:
            if "age" in filtered_df.columns:
                filtered_df["age_group"] = pd.cut(
                    filtered_df["age"],
                    bins=[0, 30, 40, 50, 60, 100],
                    labels=["30以下", "31-40", "41-50", "51-60", "60以上"]
                )

                age_churn = filtered_df.groupby("age_group", observed=False)["exited"].mean()

                fig, ax = plt.subplots()
                ax.bar(age_churn.index.astype(str), age_churn.values)
                ax.set_xlabel("Age Group")
                ax.set_ylabel("Churn Rate")
                ax.set_title("Churn Rate by Age Group")
                st.pyplot(fig)

        with col2:
            if "geography" in filtered_df.columns:
                geo_churn = filtered_df.groupby("geography")["exited"].mean()

                fig, ax = plt.subplots()
                ax.bar(geo_churn.index.astype(str), geo_churn.values)
                ax.set_xlabel("Region")
                ax.set_ylabel("Churn Rate")
                ax.set_title("Churn Rate by Region")
                st.pyplot(fig)

        col3, col4 = st.columns(2)

        with col3:
            if "numofproducts" in filtered_df.columns:
                product_churn = filtered_df.groupby("numofproducts")["exited"].mean()

                fig, ax = plt.subplots()
                ax.bar(product_churn.index.astype(str), product_churn.values)
                ax.set_xlabel("Number of Products")
                ax.set_ylabel("Churn Rate")
                ax.set_title("Churn Rate by Number of Products")
                st.pyplot(fig)

        with col4:
            if "isactivemember" in filtered_df.columns:
                active_churn = filtered_df.groupby("isactivemember")["exited"].mean()

                fig, ax = plt.subplots()
                ax.bar(active_churn.index.astype(str), active_churn.values)
                ax.set_xlabel("Active Member")
                ax.set_ylabel("Churn Rate")
                ax.set_title("Churn Rate by Active Status")
                st.pyplot(fig)

    else:
        st.warning("当前数据中没有找到流失标签字段，无法进行流失风险分析。")

    st.header("4. Prediction Model")

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

        col1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.3f}")
        col2.metric("Recall", f"{recall_score(y_test, y_pred):.3f}")
        col3.metric("F1-score", f"{f1_score(y_test, y_pred):.3f}")
        col4.metric("AUC", f"{roc_auc_score(y_test, y_prob):.3f}")

        st.subheader("混淆矩阵 Confusion Matrix")

        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(ax=ax)
        ax.set_title("Confusion Matrix")
        st.pyplot(fig)

        st.subheader("特征重要性 Feature Importance")

        importance = pd.DataFrame({
            "Feature": X.columns,
            "Importance": model.feature_importances_
        }).sort_values(by="Importance", ascending=False)

        st.dataframe(importance)

        fig, ax = plt.subplots()
        ax.barh(importance["Feature"], importance["Importance"])
        ax.set_xlabel("Importance")
        ax.set_title("Feature Importance")
        ax.invert_yaxis()
        st.pyplot(fig)

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

        st.header("5. High-Risk Customer List")

        st.subheader("流失风险概率分布")

        fig, ax = plt.subplots()
        ax.hist(result_df["churn_risk_probability"], bins=20)
        ax.set_xlabel("Churn Risk Probability")
        ax.set_ylabel("Number of Customers")
        ax.set_title("Churn Risk Probability Distribution")
        st.pyplot(fig)

        st.subheader("Top 10 高风险客户")

        top10 = result_df.sort_values(
            by="churn_risk_probability",
            ascending=False
        ).head(10)

        if "customerid" in top10.columns:
            x_values = top10["customerid"].astype(str)
        else:
            x_values = top10.index.astype(str)

        fig, ax = plt.subplots()
        ax.bar(x_values, top10["churn_risk_probability"])
        ax.set_xlabel("Customer ID")
        ax.set_ylabel("Churn Risk Probability")
        ax.set_title("Top 10 High-Risk Customers")
        plt.xticks(rotation=45)
        st.pyplot(fig)

        st.subheader("高风险客户识别与导出")

        risk_threshold = st.slider(
            "选择高风险阈值",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.05
        )

        top_n = st.selectbox(
            "选择导出客户数量",
            options=[10, 20, 50, 100, 200],
            index=2
        )

        high_risk = result_df[
            result_df["churn_risk_probability"] >= risk_threshold
        ].sort_values(
            by="churn_risk_probability",
            ascending=False
        ).head(top_n)

        st.write(f"当前筛选出 {len(high_risk)} 名高风险客户。")
        st.dataframe(high_risk)

        csv = high_risk.to_csv(index=False).encode("utf-8-sig")

        st.download_button(
            label="下载高风险客户名单 CSV",
            data=csv,
            file_name="high_risk_customers.csv",
            mime="text/csv"
        )

        st.subheader("客户分层分析")

        if "customer_segment" in result_df.columns:
            segment_count = result_df["customer_segment"].value_counts()

            col1, col2 = st.columns(2)

            with col1:
                segment_table = segment_count.reset_index()
                segment_table.columns = ["客户类型", "客户数量"]
                st.dataframe(segment_table)

            with col2:
                fig, ax = plt.subplots()
                ax.bar(segment_count.index.astype(str), segment_count.values)
                ax.set_xlabel("Customer Segment")
                ax.set_ylabel("Number of Customers")
                ax.set_title("Customer Segmentation")
                plt.xticks(rotation=30)
                st.pyplot(fig)

        st.header("6. Business Recommendations")

        st.write("本系统根据客户流失概率、账户余额、活跃状态、产品数量和年龄生成客户维护建议。")

        st.write("1. 对高风险高价值客户，优先安排客户经理回访。")
        st.write("2. 对高风险但不活跃客户，建议发送激活活动或优惠信息。")
        st.write("3. 对仅持有少量产品的高风险客户，建议进行交叉销售。")
        st.write("4. 对年龄较大的高风险客户，建议提供电话客服或线下服务支持。")
        st.write("5. 对低风险高价值客户，建议继续保持长期关系维护。")

    else:
        st.warning("当前数据中没有找到流失标签字段，无法训练预测模型。")

else:
    st.info("请先上传 CSV 文件。")

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score


st.set_page_config(
    page_title="银行客户画像与流失风险分析系统",
    layout="wide"
)

st.title("银行客户画像与流失风险分析系统")
st.write("本系统基于银行客户数据，支持客户画像分析、客户分层、流失风险识别与业务建议生成。")

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


def add_customer_segment(df):
    df = df.copy()

    balance_threshold = df["balance"].quantile(0.75) if "balance" in df.columns else 0
    salary_threshold = df["estimatedsalary"].quantile(0.75) if "estimatedsalary" in df.columns else 0

    def segment_customer(row):
        if "exited" in df.columns and row.get("exited", 0) == 1:
            return "已流失客户"

        if (
            row.get("balance", 0) >= balance_threshold
            and row.get("numofproducts", 0) >= 2
            and row.get("isactivemember", 0) == 1
        ):
            return "高价值客户"

        if (
            row.get("isactivemember", 1) == 0
            or row.get("numofproducts", 0) <= 1
        ):
            return "潜在流失客户"

        if row.get("estimatedsalary", 0) >= salary_threshold:
            return "高收入客户"

        return "普通客户"

    df["customer_segment"] = df.apply(segment_customer, axis=1)
    return df


if uploaded_file is not None:
    df = load_and_prepare_data(uploaded_file)

    if "balance" in df.columns and "numofproducts" in df.columns and "isactivemember" in df.columns:
        df = add_customer_segment(df)

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

    if "customer_segment" in df.columns:
        selected_segment = st.sidebar.multiselect(
            "客户分层",
            options=sorted(df["customer_segment"].dropna().unique()),
            default=sorted(df["customer_segment"].dropna().unique())
        )
        filtered_df = filtered_df[filtered_df["customer_segment"].isin(selected_segment)]

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

    st.subheader("客户分层分析")

    if "customer_segment" in filtered_df.columns:
        segment_count = filtered_df["customer_segment"].value_counts()

        col1, col2 = st.columns(2)

        with col1:
            st.dataframe(segment_count.reset_index().rename(
                columns={"index": "客户类型", "customer_segment": "客户数量"}
            ))

        with col2:
            fig, ax = plt.subplots()
            ax.bar(segment_count.index.astype(str), segment_count.values)
            ax.set_xlabel("Customer Segment")
            ax.set_ylabel("Number of Customers")
            ax.set_title("Customer Segmentation")
            plt.xticks(rotation=30)
            st.pyplot(fig)

    st.subheader("客户画像分析")

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

    st.subheader("流失风险分析")

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

    st.subheader("流失预测模型")

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

        st.subheader("特征重要性")

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

        st.subheader("高风险客户识别与导出")

        result_df = df.copy()

        predict_df = model_df.drop(columns=["exited"])
        result_df = result_df.loc[model_df.index]
        result_df["churn_risk_probability"] = model.predict_proba(predict_df)[:, 1]

        high_risk = result_df.sort_values(
            by="churn_risk_probability",
            ascending=False
        ).head(50)

        st.dataframe(high_risk)

        csv = high_risk.to_csv(index=False).encode("utf-8-sig")

        st.download_button(
            label="下载高风险客户名单 CSV",
            data=csv,
            file_name="high_risk_customers.csv",
            mime="text/csv"
        )

        st.subheader("业务建议")

        st.write("1. 对流失风险较高的客户进行客户经理回访或专属服务推荐。")
        st.write("2. 对高余额但不活跃客户进行重点维护，降低潜在流失风险。")
        st.write("3. 对仅持有少量产品的客户进行交叉销售，提高客户粘性。")
        st.write("4. 根据不同年龄段和地区的流失率差异，制定差异化客户维护策略。")

    else:
        st.warning("当前数据中没有找到流失标签字段，无法训练预测模型。")

else:
    st.info("请先上传 CSV 文件。")
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

# Judul aplikasi
st.title("Analisis Faktor Indikator Sosial Ekonomi untuk Pemetaan Wilayah Prioritas Bantuan Sosial di Indonesia")

# 1. Input Dataset
st.header("1. Menginputkan Data")
uploaded_file = st.file_uploader("Upload File CSV", type=["csv"])

if uploaded_file:
    try:
        # Membaca dataset
        data = pd.read_csv(uploaded_file)
        data.columns = data.columns.str.strip()  # Membersihkan nama kolom
        st.write("Dataset yang Diunggah:")
        st.dataframe(data)

        # Validasi kolom yang diperlukan
        required_columns = [
            "Rata-rata Lama Sekolah Penduduk 15+ (Tahun)",
            "Pengeluaran per Kapita Disesuaikan (Ribu Rupiah/Orang/Tahun)",
            "Indeks Pembangunan Manusia",
            "Umur harapan hidup",
            "Persentase rumah tangga yang memiliki akses terhadap sanitasi layak",
            "Persentase rumah tangga yang memiliki akses terhadap air minum layak",
            "Tingkat Pengangguran Terbuka",
            "Tingkat Partisipasi Angkatan Kerja",
            "PDRB atas Dasar Harga Konstan menurut Pengeluaran (Rupiah)",
            "Klasifikasi Kemiskinan"  # Target
        ]

        if not all(col in data.columns for col in required_columns):
            st.error("Dataset yang diunggah tidak memiliki semua kolom yang diperlukan.")
        else:
            # Preprocessing Data
            st.header("2. Preprocessing Data")
            try:
                # Penyesuaian nilai pada kolom 'Rata-rata Lama Sekolah Penduduk 15+ (Tahun)'
                column_to_adjust = "Rata-rata Lama Sekolah Penduduk 15+ (Tahun)"
                if column_to_adjust in data.columns:
                    data[column_to_adjust] = data[column_to_adjust].apply(
                        lambda x: round(x / 100, 2) if x > 20 else round(x, 2)
                    )
                    st.info(f"Nilai di kolom '{column_to_adjust}' yang lebih dari 20.00 telah diubah menjadi desimal dengan 2 angka di belakang koma.")
                else:
                    st.warning(f"Kolom '{column_to_adjust}' tidak ditemukan dalam dataset.")

                # Validasi ulang kolom yang diperlukan
                required_columns = [
                    "Rata-rata Lama Sekolah Penduduk 15+ (Tahun)",
                    "Pengeluaran per Kapita Disesuaikan (Ribu Rupiah/Orang/Tahun)",
                    "Indeks Pembangunan Manusia",
                    "Umur harapan hidup",
                    "Persentase rumah tangga yang memiliki akses terhadap sanitasi layak",
                    "Persentase rumah tangga yang memiliki akses terhadap air minum layak",
                    "Tingkat Pengangguran Terbuka",
                    "Tingkat Partisipasi Angkatan Kerja",
                    "PDRB atas Dasar Harga Konstan menurut Pengeluaran (Rupiah)",
                    "Klasifikasi Kemiskinan"  # Target
                ]

                if not all(col in data.columns for col in required_columns):
                    st.error("Dataset yang diunggah tidak memiliki semua kolom yang diperlukan setelah preprocessing.")
                else:
                    data = data.dropna().drop_duplicates()
                    st.subheader("Dataset Setelah Preprocessing (Menghapus Duplikasi dan Nilai Kosong)")
                    st.dataframe(data)

                    # Pisahkan fitur (X) dan target (y)
                    X = data[required_columns[:-1]]
                    y = data["Klasifikasi Kemiskinan"]

                    # Standarisasi fitur
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    X_scaled_df = pd.DataFrame(X_scaled, columns=required_columns[:-1])

                    st.subheader("Data Setelah Standarisasi")
                    st.dataframe(X_scaled_df)

                    # Split dataset
                    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

            except KeyError as e:
                st.error(f"Terjadi kesalahan: {e}")

            # 3. Analisis Data dan Model Random Forest
            st.header("3. Analisis Data dan Model Random Forest")
            rf_model = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=10)
            rf_model.fit(X_train, y_train)
            y_pred = rf_model.predict(X_test)

            # Evaluasi Model
            st.subheader("Hasil Evaluasi Model")
            classification_rep = classification_report(y_test, y_pred, output_dict=True)
            st.write(pd.DataFrame(classification_rep).transpose())

            # Confusion Matrix
            st.subheader("Confusion Matrix")
            conf_matrix = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
            plt.xlabel("Prediksi")
            plt.ylabel("Aktual")
            st.pyplot(fig)

            # ROC Curve
            st.subheader("ROC Curve")
            if len(set(y)) == 2:  # ROC hanya untuk binary classification
                y_prob = rf_model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                auc = roc_auc_score(y_test, y_prob)

                fig, ax = plt.subplots()
                plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
                plt.plot([0, 1], [0, 1], 'r--')
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.title("ROC Curve")
                plt.legend()
                st.pyplot(fig)
            else:
                st.info("ROC Curve hanya tersedia untuk klasifikasi biner.")

            # Feature Importance
            st.subheader("Feature Importance")
            feature_importance = pd.DataFrame({
                "Fitur": required_columns[:-1],
                "Pentingnya": rf_model.feature_importances_
            }).sort_values(by="Pentingnya", ascending=False)

            st.write(feature_importance)
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(data=feature_importance[:10], x="Pentingnya", y="Fitur", palette="viridis")
            plt.title("10 Fitur Terpenting")
            st.pyplot(fig)

            # Prediksi Berdasarkan Input Manual
            st.header("Prediksi Berdasarkan Input Manual")
            manual_input = {}
            for col in required_columns[:-1]:
                manual_input[col] = st.number_input(f"Masukkan nilai untuk {col}", step=0.01)

            if st.button("Prediksi Berdasarkan Input"):
                input_df = pd.DataFrame([manual_input])
                input_scaled = scaler.transform(input_df)
                manual_prediction = rf_model.predict(input_scaled)[0]
                st.success(f"Hasil Prediksi: {manual_prediction}")

            # 9. Tambahan Analisis Statistik Deskriptif
            st.header("Statistik Deskriptif Dataset")
            if "data" in locals():
                st.subheader("Ringkasan Statistik")
                st.write(data.describe())
            else:
                st.warning("Dataset belum tersedia. Silakan unggah dataset terlebih dahulu.")

            # Visualisasi Data Berdasarkan Provinsi dan Indikator 
            st.header("Visualisasi Data Berdasarkan Provinsi dan Indikator")

            if "Provinsi" not in data.columns:
                st.error("Kolom 'Provinsi' tidak ditemukan dalam dataset. Pastikan dataset memiliki kolom ini.")
            else:
                # Pilih variabel untuk sumbu Y
                y_col = st.selectbox(
                    "Pilih Indikator untuk Sumbu Y :",
                    [
                        "Rata-rata Lama Sekolah Penduduk 15+ (Tahun)",
                        "Pengeluaran per Kapita Disesuaikan (Ribu Rupiah/Orang/Tahun)",
                        "Indeks Pembangunan Manusia",
                        "Umur harapan hidup",
                        "Persentase rumah tangga yang memiliki akses terhadap sanitasi layak",
                        "Persentase rumah tangga yang memiliki akses terhadap air minum layak",
                        "Tingkat Pengangguran Terbuka",
                        "Tingkat Partisipasi Angkatan Kerja",
                        "PDRB atas Dasar Harga Konstan menurut Pengeluaran (Rupiah)"
                    ]
                )

                # Pilih bentuk visualisasi
                chart_type = st.selectbox(
                    "Pilih Bentuk Visualisasi:",
                    ["Bar Plot", "Line Plot", "Scatter Plot"]
                )

                if y_col in data.columns:
                    # Agregasi data (rata-rata per provinsi)
                    provinsi_data = data.groupby("Provinsi")[y_col].mean().reset_index()

                    # Membuat visualisasi data
                    st.subheader(f"Visualisasi Data: {y_col} berdasarkan Provinsi")

                    fig, ax = plt.subplots(figsize=(14, 8))
                    if chart_type == "Bar Plot":
                        sns.barplot(data=provinsi_data, x="Provinsi", y=y_col, palette="viridis", ci=None, ax=ax)
                    elif chart_type == "Line Plot":
                        sns.lineplot(data=provinsi_data, x="Provinsi", y=y_col, marker="o", ax=ax)
                    elif chart_type == "Scatter Plot":
                        sns.scatterplot(data=provinsi_data, x="Provinsi", y=y_col, s=100, ax=ax)

                    plt.xticks(rotation=45, ha="right")
                    plt.xlabel("Provinsi")
                    plt.ylabel(y_col)
                    plt.title(f"{chart_type} {y_col} Berdasarkan Provinsi")
                    st.pyplot(fig)

                    # Statistik deskriptif untuk indikator
                    st.subheader("Statistik Deskriptif untuk Indikator yang Dipilih")
                    st.write(provinsi_data.describe())
                else:
                    st.warning(f"Kolom '{y_col}' tidak ditemukan")

    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")
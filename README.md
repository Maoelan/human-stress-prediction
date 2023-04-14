# Submission 1: Machine Learning Pipeline - Human Stress Prediction
Nama: Maulana Muhammad

Username dicoding: maoelana

| | Deskripsi |
| ----------- | ----------- |
| Dataset | [Human Stress Prediction](https://www.kaggle.com/datasets/kreeshrajani/human-stress-prediction) |
| Masalah | Stres adalah reaksi seseorang baik secara fisik maupun emosional (mental/psikis) apabila ada perubahan dari lingkungan yang mengharuskan seseorang menyesuaikan diri. Stres adalah bagian alami dan penting dari kehidupan, tetapi apabila berat dan berlangsung lama dapat merusak kesehatan |
| Solusi machine learning | Stress susah dilihat dari keseharian seseorang, oleh karena itu dengan machine learning dapat mengetahui apakah seseorang stress hanya dari ketikan seseorang yang dilakukan dimedia sosial |
| Metode pengolahan | Pada data Human Stress Prediction, terdapat tujuh feature, tetapi yang digunakan pada proyek ini hanya feature text dan label, sehingga features selain itu akan dihapus, kemudian dilakukan split data training dan eval menjadi rasio 80:20, dan mengubah data feature menjadi lowercase serta feature label menjadi integer |
| Arsitektur model | Arsitektur model yang digunakan yaitu model embedding dimana terdiri dari vectorize_layer, kemudian layer embedding dengan dimensi embedding yaitu 16, setelah itu layer AveragePooling1D karena data merupakan bentuk text, kemudian layer dense 64, 32 dengan activation relu dan sigmoid karena akan dilakukan klasifikasi antar dua label. Loss yang digunakan binary_crossentropy dengan optimizer Adam dan metrik BinaryAccuray |
| Metrik evaluasi | Metrik evaluasi yang digunakan yaitu ExampleCount, AUC, FalsePositives, TruePositives, FalseNegatives, TrueNegatives, dan BinaryAccuracy |
| Performa model | Evaluasi model diperoleh yaitu AUC sebesar 82%, kemudian example_count 575, dengan BinaryAccuracy 75%, dan loss sebesar 1.364. Untuk False Negatives 68, False Positive 75, True Negative 201 dan True Positive 231. Model yang telah dibuat dapat dilakukan peningkatan performa, karena model belum cukup baik karena BinaryAccuracy masih dibawah 80% |

import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC


ana_klasor = "/content/drive/MyDrive/metinler"


def yukle_kose_yazilari(ana_klasor):
    kose_yazilari = []
    yazarlar = []
    for dosya_veya_klasor in os.listdir(ana_klasor):  
        dosya_veya_klasor_yolu = os.path.join(ana_klasor, dosya_veya_klasor)  
        if os.path.isdir(dosya_veya_klasor_yolu): 
            for dosya in os.listdir(dosya_veya_klasor_yolu): 
                with open(os.path.join(dosya_veya_klasor_yolu, dosya), "r", encoding="utf-8") as f:
                    kose_yazilari.append(f.read())
                    yazarlar.append(dosya_veya_klasor) 
    return kose_yazilari, yazarlar


kose_yazilari, yazarlar = yukle_kose_yazilari(ana_klasor)
veri_seti = pd.DataFrame({
    'Köşe_Yazısı': kose_yazilari,
    'Yazar': yazarlar
})


tfidf_vectorizer = TfidfVectorizer()
X = tfidf_vectorizer.fit_transform(veri_seti['Köşe_Yazısı'])


model = LinearSVC()
model.fit(X, veri_seti['Yazar'])


def yazar_tahmini(metin):
    metin_vektoru = tfidf_vectorizer.transform([metin])
    tahmin = model.predict(metin_vektoru)
    return tahmin[0]


test_metni = "Sayın Bakan'ın, PCR testi ile ilgili kararı sürpriz olmadı. Aşı karşıtları hemen bu kararların kendi mücadelelerinin başarısı olduğunu ilan ettiler. Ama işin aslı öyle değil. Bu kararların arkasındaki neden artık bakanlığın katı devletçi PCR test politikasının sürdürülebilir olamaması. Daha öncede yazmıştım. Halen bakanlığın tek tanı yöntemini PCR testi ve bu testin de sadece hastanelerde yapılabilir olması kararı sonuçta hastalık semptomları gösteren insanları ya devlet hastanelerinde ya da özel laboratuvarlarda ciddi paralar vererek test yaptırmaya zorlayan bir uygulama. ."
yazar = yazar_tahmini(test_metni)
print("Metin hangi yazardan:", yazar)

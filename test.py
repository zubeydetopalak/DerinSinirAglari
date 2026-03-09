import numpy as np
import pickle
import os


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

print("Lokal veriler yükleniyor...")
veri_yolu = 'cifar-10-batches-py'


train_batch = unpickle(os.path.join(veri_yolu, 'data_batch_1'))
X_train = train_batch[b'data'].astype('int32')
Y_train = np.array(train_batch[b'labels'])

test_batch = unpickle(os.path.join(veri_yolu, 'test_batch'))
X_test = test_batch[b'data'].astype('int32')
Y_test = np.array(test_batch[b'labels'])

meta = unpickle(os.path.join(veri_yolu, 'batches.meta'))
sinif_isimleri = [label.decode('utf-8') for label in meta[b'label_names']]



print("-" * 30)
mesafe_tipi = input("Hangi mesafe ölçümü kullanılsın? (L1 / L2): ").strip().upper()
k_degeri = int(input("k komşu değerini girin: "))



test_edilecek_resim_sayisi = 1000
dogru_tahmin_sayisi = 0

print("-" * 30)
print(f"İlk {test_edilecek_resim_sayisi} test resmi için {mesafe_tipi} mesafesi ve k={k_degeri} ile hesaplanıyor...")
print("DİKKAT: Bu işlem 1000x10.000 hesaplama yapacağı için yaklaşık 1-2 dakika sürebilir.")
print("Lütfen bekleyin, işlem durumu aşağıda güncellenecektir...\n")

for i in range(test_edilecek_resim_sayisi):
    test_resmi = X_test[i]
    gercek_etiket = Y_test[i]
    
    
    if mesafe_tipi == 'L1':
        mesafeler = np.sum(np.abs(X_train - test_resmi), axis=1)
    elif mesafe_tipi == 'L2':
        mesafeler = np.sqrt(np.sum(np.square(X_train - test_resmi), axis=1))
    else:
        print("\nHatalı giriş yaptınız. Lütfen 'L1' veya 'L2' yazın.")
        exit()
        
    
    sirali_indeksler = np.argsort(mesafeler)
    en_yakin_indeksler = sirali_indeksler[:k_degeri]
    k_yakin_etiketler = Y_train[en_yakin_indeksler]
    
   
    benzersiz_etiketler, tekrar_sayilari = np.unique(k_yakin_etiketler, return_counts=True)
    en_cok_gecen_indeks = np.argmax(tekrar_sayilari)
    tahmin_edilen_etiket = benzersiz_etiketler[en_cok_gecen_indeks]
    
    
    if tahmin_edilen_etiket == gercek_etiket:
        dogru_tahmin_sayisi += 1
        
   
    if (i + 1) % 100 == 0:
        print(f"İlerleme: {i + 1} / {test_edilecek_resim_sayisi} resim tamamlandı...")


basari_yuzdesi = (dogru_tahmin_sayisi / test_edilecek_resim_sayisi) * 100

print("\n" + "=" * 30)
print("TEST BAŞARIYLA TAMAMLANDI!")
print(f"Toplam Test Edilen Resim: {test_edilecek_resim_sayisi}")
print(f"Doğru Bilinen Resim: {dogru_tahmin_sayisi}")
print(f"Yanlış Bilinen Resim: {test_edilecek_resim_sayisi - dogru_tahmin_sayisi}")
print(f"Kullanılan Mesafe Ölçümü: {mesafe_tipi}")
print(f"Kullanılan k değeri: {k_degeri}")
print(f"--> Modelin Başarı Oranı (Accuracy): %{basari_yuzdesi:.2f}")
print("=" * 30)

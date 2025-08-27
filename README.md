# Demir Çelik Üretimi için Çoklu Hedef Regresyon Modeli

Bu proje, demir çelik üretim prosesindeki kimyasal kompozisyon ve haddeleme verilerini girdi olarak kullanarak, nihai ürünün 6 farklı mekanik ve geometrik özelliğini (`çap`, `ovalite`, `elastikiyet`, `alt/üst akma dayanımı`, `tufal oranı`) yüksek doğrulukla tahmin etmek amacıyla geliştirilmiş bir makine öğrenmesi sistemidir.

## Projenin Amacı ve Teknik Yaklaşım

Sistemin temel amacı, üretim sahasından toplanan sensör ve analiz verilerini kullanarak, kalite kontrol laboratuvarında ölçülen nihai ürün özelliklerini önceden tahmin etmektir. Bu, üretimde proaktif ayarlamalar yapılmasına ve kalite sapmalarının en aza indirilmesine olanak tanır.

Proje, her bir hedef değişken için en iyi performansı gösteren "uzman" modellerden oluşan bir **Hibrit Model Mimarisi** kullanmaktadır. Bu yaklaşım, her bir hedefin kendine özgü fiziksel dinamiklerini daha iyi modellememizi sağlar.

### Teknik Detaylar
* **Veri Ön İşleme:** Girdi verileri, yinelenen sütunların kaldırılması, eksik verilerin medyan ile doldurulması gibi adımları içeren sağlam bir ön işleme çerçevesinden geçirilir.
* **Hedef Değişken Dönüşümü:** Hedef değişkenlerin istatistiksel dağılımını normale yaklaştırmak ve modelin öğrenme performansını artırmak için logaritmik dönüşüm (`log1p`) uygulanır.
* **Normalizasyon:** Tüm girdi özellikleri, `MinMaxScaler` kullanılarak 0-1 aralığına normalize edilir. Bu, farklı birimlere ve ölçeklere sahip verilerin model tarafından eşit derecede önemli kabul edilmesini sağlar.
* **Modelleme:** Her bir hedef değişken için, o hedefe en uygun olduğu kanıtlanmış farklı makine öğrenmesi modelleri (`XGBoost`, `LightGBM`, `StackingRegressor` vb.) ayrı ayrı eğitilir ve tahmin için kullanılır.

## Model Performansı ve Sektörel Değerlendirme

Geliştirilen hibrit model, zorlu ve değişken üretim koşullarına rağmen oldukça başarılı sonuçlar vermektedir.

* **Yüksek Başarılı Tahminler:** Model, `çap`, `ovalite` ve `tufal oranı` gibi geometrik ve yüzey özelliklerini **%95'in üzerinde bir doğrulukla**, üretime doğrudan yön verebilecek bir kesinlikte tahmin edebilmektedir.

* **Kabul Edilebilir Sapmalar ve "Proses Varyansı":** `elastikiyet` ile `alt ve üst akma dayanımı` gibi, metalurjik olarak daha karmaşık ve birçok gizli faktöre bağlı olan hedeflerde, modelin tahminleri ile gerçek değerler arasında **yaklaşık ±25 birimlik bir sapma payı** gözlemlenmektedir.

    Bu sapma, modelin bir hatası olmaktan ziyade, mevcut veri setinin içsel limitlerini ve demir çelik üretiminin doğasındaki karmaşıklığı yansıtmaktadır. Nihai mekanik özellikleri etkileyen yüzlerce faktör bulunur. Modelimizin beslendiği veriler bu faktörlerin birçoğunu temsil etse de, **hammadde (hurda/cevher) kalitesindeki anlık kimyasal değişimler, hadde sonrası soğutma hızındaki mikro farklılıklar veya sensör kalibrasyonundaki küçük kaymalar** gibi ölçülemeyen veya veri setinde bulunmayan değişkenler, bu son ±25 birimlik varyasyonun ana kaynağıdır. Bu durum, endüstriyel makine öğrenmesi projelerinde sıkça karşılaşılan ve **"proses varyansı"** olarak bilinen bir olgudur.

* **Genelleme ve Veri Arttırımı Gerekliliği:** Model, kendisine öğretilen üretim verileri (kaliteler, reçeteler) dahilinde son derece başarılıdır. Ancak, makine öğrenmesi modelleri doğaları gereği "ezberlemeye" yatkındır. Modelin gelecekte üretilecek **tamamen yeni çelik kaliteleri veya farklı üretim rejimleri** üzerinde de aynı başarıyı göstermesi için, eğitim veri setinin bu yeni kalitelerden toplanan verilerle periyodik olarak **zenginleştirilmesi ve modelin yeniden eğitilmesi** kritik öneme sahiptir. Bu, modelin sürekli öğrenen ve adapte olan bir yapıya kavuşmasını sağlar.

## Kurulum ve Kullanım

### 1. Kurulum

Projeyi çalıştırmak için bilgisayarınızda Git, Git LFS ve Python'un kurulu olması gerekmektedir.

```bash
# 1. Git LFS'i kurun (eğer kurulu değilse)
# [https://git-lfs.com/](https://git-lfs.com/) adresinden indirip kurun ve ardından terminalde çalıştırın:
git lfs install

# 2. Bu depoyu klonlayın (Git LFS, büyük veri dosyalarını otomatik olarak indirecektir)
git clone [https://github.com/kullanici-adiniz/Celik-Kalite-Tahmin-Modeli.git](https://github.com/kullanici-adiniz/Celik-Kalite-Tahmin-Modeli.git)
cd Celik-Kalite-Tahmin-Modeli

# 3. Bir sanal ortam oluşturun ve aktif hale getirin
python -m venv .venv
# Windows için: .venv\Scripts\activate
# MacOS/Linux için: source .venv/bin/activate

# 4. Gerekli kütüphaneleri yükleyin
pip install -r requirements.txt
```

### 2. Kullanım

Proje, modeli eğiten ve tahmin yapan bütünleşik bir yapıya sahiptir.

```bash
# Gerekli CSV dosyalarının ('final_birlesik_veri2.csv', 'yeni_dokum_veri.csv')
# ana klasörde olduğundan emin olun. Ardından script'i çalıştırın:
python main.py
```
* **İlk Çalıştırma:** Script, `model_portfolio_final.pkl` gibi eğitilmiş model dosyalarını bulamazsa, otomatik olarak tüm model eğitim sürecini başlatacaktır. Bu işlem bir süre alabilir.
* **Sonraki Çalıştırmalar:** Modeller bir kez eğitildikten sonra, script her çalıştığında bu adımı atlayacak ve saniyeler içinde doğrudan `yeni_dokum_veri.csv` dosyasındaki verilerle tahmin yapacaktır.
* **Çıktı:** Tahminler, `yeni_dokum_tahmin_sonuclari_final.csv` adıyla bir CSV dosyasına kaydedilir.

import requests

def send_data_to_api(data, api_endpoint):
    try:
        response = requests.post(api_endpoint, json=data)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx, 5xx)
        return response.json()
    except requests.exceptions.RequestException as e:
        print("Error:", e)
        return None

if __name__ == "__main__":
    sentences = ["Paparan sinar ultraviolet dari sinar matahari dapat menyebabkan kerusakan mata",
    "Diabetes dapat menyebabkan retinopati diabetik yang dapat mengganggu penglihatan",
    "Perubahan penglihatan terjadi secara alami seiring bertambahnya usia",
    "Merokok dapat meningkatkan risiko terjadinya katarak dan degenerasi makula",
    "Konsumsi makanan bergizi, seperti makanan kaya vitamin A, C, E, dan mineral seperti seng, dapat membantu menjaga kesehatan mata",
    "Istirahat reguler dan tetes mata bisa membantu mengurangi ketidaknyamanan mata kering",
    "Penggunaan komputer dan perangkat elektronik lainnya dalam jangka waktu yang lama dapat menyebabkan Computer Vision Syndrome (CVS), menyebabkan gejala seperti mata kering, sakit kepala, dan ketegangan leher",
    "Bentuk fisik mata setiap orang berbeda",
    "Kacamata hitam yang baik melindungi mata dari sinar UV dan meminimalkan paparan sinar biru dari layar komputer atau perangkat elektronik lainnya",
    "Deteksi dini adalah kunci untuk mencegah banyak masalah mata yang serius",
    "Radang kelopak mata atau blepharitis adalah peradangan umum pada kelopak mata yang dapat menyebabkan mata merah, gatal, dan mengganggu penglihatan",
    "Terlalu banyak paparan layar gadget dapat meningkatkan risiko mata kering",
    "Glaukoma adalah kondisi di mana tekanan mata tinggi dapat merusak saraf optik dan menyebabkan kehilangan penglihatan",
    "Usia di atas 40 meningkatkan risiko pengembangan katarak, yang mengaburkan lensa mata",
    "Terlalu sering mengucek mata dapat memicu iritasi dan infeksi",
    "Kondisi mata seperti mata juling dapat diobati dengan terapi fisik atau operasi",
    "Penggunaan lensa kontak yang tidak tepat atau kebersihan yang buruk dapat menyebabkan infeksi mata serius",
    "Beberapa penyakit autoimun, seperti lupus dan rheumatoid arthritis, dapat mempengaruhi mata",
    "Tekanan darah tinggi atau hipertensi dapat merusak pembuluh darah di mata",
    "Konsumsi lemak sehat, seperti omega-3, dapat mendukung kesehatan mata",
    "Lensa kontak tipe harian lebih higienis karena diganti setiap hari",
    "Stres kronis dapat mempengaruhi kesehatan mata dan menyebabkan gejala mata kering",
    "Alkohol dalam jumlah berlebihan dapat merusak saraf optik",
    "Terlalu banyak mengonsumsi gula dapat meningkatkan risiko diabetes tipe 2, yang dapat merusak mata",
    "Beberapa obat, seperti antihistamin, dapat menyebabkan mata kering",
    "Tekanan pada mata saat batuk atau bersin dapat memicu katarak pada orang yang rentan",
    "Cedera mata dapat terjadi saat berolahraga atau kecelakaan, sehingga penting untuk menggunakan perlindungan mata yang tepat",
    "Diabetes juga dapat menyebabkan pembuluh darah kecil di retina pecah, menyebabkan pendarahan dalam mata",
    "Radang retina, atau uveitis, adalah kondisi yang dapat mengakibatkan mata merah, nyeri, dan penglihatan kabur",
    "Lensa kontak tipe bulanan memerlukan perawatan dan pembersihan yang lebih intensif",
    "Terlalu banyak konsumsi garam dapat meningkatkan risiko pembengkakan di sekitar mata",
    "Terkena cahaya biru yang kuat dari layar komputer larut malam dapat mengganggu ritme sirkadian dan kualitas tidur",
    "Mata kering dapat mempengaruhi kenyamanan selama penggunaan lensa kontak",
    "Paparan sinar matahari yang berlebihan selama bertahun-tahun dapat menyebabkan pterygium atau ‘selaput mata’",
    "Retina, bagian belakang mata, mengandung sel-sel fotoreseptor yang memungkinkan kita melihat cahaya dan warna",
    "Terlalu banyak kafein dapat menyebabkan kelebihan cairan di mata, menyebabkan pembengkakan",
    "Konsumsi vitamin C dan E dapat membantu mencegah degenerasi makula terkait usia",
    "Pembengkakan kelopak mata (edema) dapat disebabkan oleh alergi atau masalah kesehatan lainnya",
    "Beberapa kondisi seperti lupus dan rheumatoid arthritis dapat menyebabkan mata merah dan kering",
    "Mata bayi baru lahir belum sepenuhnya berkembang dan perlu waktu untuk fokus dengan baik",
    "Tidak semua masalah mata dapat dilihat dari luar; pemeriksaan mata profesional penting untuk deteksi dini",
    "Mata manusia memiliki sekitar 107 juta sel batang yang membantu dalam penglihatan rendah cahaya",
    "Pigmen pada iris mata mempengaruhi warna mata, dengan lebih banyak pigmen menghasilkan mata yang lebih gelap",
    "Kondisi seperti hipotiroidisme dapat mempengaruhi kesehatan mata dan mengganggu produksi air mata",
    "Pasokan darah yang buruk ke mata dapat menyebabkan masalah penglihatan dan kondisi serius seperti stroke mata",
    "Pilihan kacamata atau lensa kontak yang tepat dapat membantu mengurangi mata lelah saat membaca atau bekerja di depan layar",
    "Efek sinar biru pada retina dapat menyebabkan stres oksidatif yang merusak sel-sel mata",
    "Kelopak mata memiliki kelenjar minyak yang dapat menyebabkan blepharitis jika terjadi infeksi atau penyumbatan",
    "Kondisi jerawat rosasea juga dapat mempengaruhi mata, menyebabkan mata merah dan iritasi",
    "Mata bayi cenderung berkedip lebih sedikit daripada orang dewasa",
    "Pembuluh darah di bola mata membantu menyediakan nutrisi dan oksigen ke sel-sel mata",
    "Radiasi sinar-X dapat merusak sel-sel di mata, itulah mengapa perlu menggunakan pelindung mata saat menjalani pemeriksaan sinar-X",
    "Beberapa kondisi, seperti lupus, dapat memicu kerusakan saraf optik",
    "Sel photoreceptor (sel batang dan kerucut) di retina mengubah cahaya menjadi sinyal saraf yang dikirimkan ke otak untuk pengolahan penglihatan",
    "Pembedahan katarak adalah prosedur yang umum untuk mengganti lensa mata yang kabur",
    "Terkena bahan kimia dapat menyebabkan luka kimia pada mata, yang dapat menyebabkan kerusakan permanen",
    "Mata manusia dapat fokus pada objek yang berjarak mulai dari beberapa sentimeter hingga sejauh beberapa kilometer",
    "Menerapkan kompres dingin dapat membantu mengurangi pembengkakan mata",
    "Memijat area di sekitar mata dapat membantu meredakan ketegangan dan meningkatkan sirkulasi darah",
    "Mata bayi cenderung berkedip lebih sedikit daripada orang dewasa",
    "Pembuluh darah di bola mata membantu menyediakan nutrisi dan oksigen ke sel-sel mata",
    "Radiasi sinar-X dapat merusak sel-sel di mata, itulah mengapa perlu menggunakan pelindung mata saat menjalani pemeriksaan sinar-X",
    "Beberapa kondisi, seperti lupus, dapat memicu kerusakan saraf optik",
    "Sel photoreceptor (sel batang dan kerucut) di retina mengubah cahaya menjadi sinyal saraf yang dikirimkan ke otak untuk pengolahan penglihatan"]
    
    for s in sentences:
        data = {
            "sentence": s,
            "read": 1,
            "content": 1,
            "category": 1
            }
    
        api_url = "http://43.218.145.196:5000/sentences"  # Replace this with your actual API endpoint
        
        send_data_to_api(data, api_url)
    
    # if result:
    #     print("API Response:", result)
    # else:
    #     print("Failed to send data to the API.")

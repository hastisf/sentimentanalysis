import streamlit as st
import joblib
import nltk
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import re
import string

# Konfigurasi Page
st.set_page_config(page_title="Analisis Sentimen M-Pajak", page_icon="logo.png", layout="wide")

# --- SETUP RESOURCES ---
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords
stop_words = set(stopwords.words('indonesian'))

# Custom Stopwords
custom_stopwords = {'yg','aja','nya','sih','nih','dong','lah','mah','kok','gak','ga','nggak','udah','sdh'}
stop_words.update(custom_stopwords)

factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Kamus Slang (Lengkap sesuai input Anda)
slang_dict = {
    'abal':'abal-abal','abalabal':'abal-abal','abdet':'update',
    'aflikasi':'aplikasi','aj':'saja','aja': 'saja', 'ajaa': 'saja',
    'ajah': 'saja', 'ajalah': 'saja', 'ajg': 'anjing', 'ajh': 'saja',
    'ajha': 'saja', 'aktik': 'aktif', 'aktifasi': 'aktivasi',
    'ampee':'sampai','ampe': 'sampai','ampas': 'ampas',
    'ampaslah': 'ampas', 'ampunnn':'ampun','ancur': 'hancur',
    'anj': 'anjing','anjengggg': 'anjing', 'anjg':'anjing','anjir': 'anjing',
    'anjggg': 'anjing',
    'anjr':'anjing','anjrit': 'anjing', 'anying': 'anjing',
    'anyinh': 'anjing','anyingg': 'anjing','anjim' : 'anjing',
    'anjay' : 'anjing', 'anjr' : 'anjing', 'anying' : 'anjing',
    'anyingg' : 'anjing','apa2':'apa apa',
    'apaansi': 'apaan sih','apk': 'aplikasi', 'apkikasinya':'aplikasinya',
    'apkn':'aplikasi','apknya':'aplikasi','aplikaainya':'aplikasinya',
    'aplikasih':'aplikasi','app': 'aplikasi','apps':'aplikasi',
    'aps':'aplikasi','awikwok':'tertawa' ,'bae':'saja',
    'baget': 'banget', 'bagus':'bagus', 'bagusan': 'bagus',
    'banyakkk': 'banyak','bangke':'banget','bapuk':'jelek',
    'berjam2': 'berjam-jam', 'berkali': 'berkali-kali',
    'berteletele': 'tidak jelas', 'bgtt': 'banget', 'bgt':'banget',
    'bgttt':'banget','bgus':'bagus','bikan':'bikin','bikin':'membuat',
    'bkin':'membuat','blm':'belum','bnget':'banget','bngtt':'banget',
    'bngung':'bingung','bnyk':'banyak','bnyak':'banyak',
   'boncos': 'rugi', 'boro2': 'boro-boro', 'br':'benar',
    'broken':'rusak','bungol': 'bodoh', 'burik':'buruk','busok':'busuk',
    'butut': 'buruk', 'byr': 'bayar','cacad': 'cacat',
    'capslock': 'caps lock', 'capecape': 'capek capek', 'cht': 'chat',
    'cipatin': 'ciptain', 'cokkkkk': 'cuk', 'coraetax': 'coretax',
    'coretak':'coretax','coretex':'coretax','cortec':'coretax',
    'cortes':'coretax','cortetak':'coretax','cortex':'coretax',
    'coretek':'coretax','coretak':'coretax','crhome':'chrome','crome':'chrome',
    'dapt':'dapat','daptar':'daftar','dateng':'datang','dbuat':'dibuat',
    'dgn':'dengan','dikirim2':'dikirim','dikirim"': 'dikirim',
    'dimasukan': 'dimasukkan','ditanyain':'ditanyakan','dongo': 'dongo',
    'dwld': 'download', 'eefin':'efin', 'efein': 'efin','effin': 'efin',
    'efiin': 'efin', 'efin': 'efin', 'efrin': 'efin', "efinnya":"efin",
    'elek': 'jelek', 'emang': 'memang','emg': 'memang', 'emprottt': 'emprot',
    'engga': 'tidak', 'enggak': 'tidak','eror': 'error', 'erro':'error',
    'erorr': 'error', 'errrror': 'error', 'errorrr': 'error', 'erroor':'error',
    "errorr":"error",'evin': 'efin', "cras":"crash", 'crashh':'crash',
    'kadaluarasa':'kedaluwarsa','expired':'kedaluwarsa',"force-close":"forceclose",
    'force close':'forceclose','g':'tidak','gagal':'gagal', 'gaje':'tidak jelas',
    'gajelas': 'tidak jelas', "kadaluarsa":"kedaluwarsa",
    'gajlss': 'tidak jelas', 'gblk': 'goblok', 'gada': 'tidak ada',
    "gagalnya":"gagal",'gae': 'membuat','gaguna': 'tidak berguna',
    'gajadi':'tidak jadi', 'fc':'forceclose','gbs': 'tidak bisa','gedeg':'kesal',
    'ghoblok': 'goblok', 'ginii': 'gini','gmn': 'bagaimana', 'gmna':'bagaimana',
    'goblk':'goblok','goblokk': 'goblok','good': 'bagus', 'gtw': 'gak tahu',
    'hacehhh': 'haduh','jdi': 'jadi', 'jelass': 'jelas','jink':'anjing',
    'josss': 'jos', 'kagak': 'tidak', 'kacao': 'kacau', 'kalo': 'kalau',
    'karna': 'karena', 'karnaa': 'karena','kek': 'kayak', 'kekick': 'keluar',
    'kemkeu': 'Kemenkeu', 'kentir': 'aneh', 'kg': 'tidak','kgk': 'tidak',
    'kl': 'kalau', 'klu': 'kalau', 'klau': 'kalau', 'kliatan': 'kelihatan',
    'knp': 'kenapa', 'kompirmasi': 'konfirmasi','konoha': 'indonesia',
    'kortek':'coretax' ,'krg':'kurang','krn': 'karena', 'kyk': 'seperti',
    'lag': 'lag', "lagg":"lag",'lemott': 'lemot','lemottt': 'lemot',
    'loginbgak':'login','loginbug':'login bug', "loginin":"login",
    'loging': 'login', 'loginnya': 'login','loginn': 'login', "logoutnya":"logout",
    'main2': 'main-main', 'maintenis': 'maintenance','makasih': 'terima kasih',
    'makasi': 'terima kasih', 'mantaffff': 'mantap', 'mantaaap': 'mantap',
    "mantaaapp":"mantap",'mantapp':'mantap','mantappp':'mantap','manthaafff':'mantap',
    'masukin': 'memasukkan','matap': 'mantap','melulu':'terus','menti':'menit',
    'mulu': 'terus', 'muluu': 'terus','muluuu': 'terus',
    'musti': 'mesti', 'naajissss': 'najis','ngebang':'hang','ngebug':'bug',
    'ngelag':'lag','ngelaggg': 'lag', 'ngga': 'tidak', 'nggak': 'tidak',
    'nggk': 'tidak','ngisinya': 'mengisinya', 'ngisi': 'mengisi',
    'nipurakyat': 'tipu rakyat', 'no': 'nomor', 'nomot': 'nomor',
    'notif': 'notifikasi', 'notivikasi': 'notifikasi', 'nyari': 'mencari',
    'nyimpen': 'menyimpan', 'nyusahin': 'menyusahkan', 'okee': 'oke',
    'orng²':'orang-orang','otw':'menuju','pajakk':'pajak','pantesan':'pantas',
    'pasword': 'password','pencet': 'tekan','pisann':'pisan',
    'plis': 'please', 'plisa': 'please', 'polri': 'polri',
    'prematur': 'belum matang', 'prik': 'aneh', 'rata2': 'rata-rata',
    'regestrasi': 'registrasi' , 'ribettt': 'ribet', 'sampe': 'sampai',
    'samsek': 'sama sekali','stuk': 'stuck', 'suport': 'support', 'sush':'susah',
    'susahh': 'susah','sussah': 'susah','taii':'tai','taiiii':'tai','tataian':'tai',
    'telfon': 'telepon','temen²':'teman-teman', 'teross':'terus','th':'tahun',
    'thn': 'tahun', 'thx': 'terima kasih','tilep': 'tilap', 'tlong': 'tolong',
    'tololl':'tolol','tolollllll':'tolol','trnyata':'ternyata',
    'trobel':'trouble','trooosss':'terus','trs':'terus','ttd':'tanda tangan',
    'tuhhh': 'tuh', 'udh': 'sudah', 'uadh': 'sudah', 'udhh': 'sudah',
    'unnistall': 'uninstall','utk': 'untuk', 'verif': 'verifikasi',
    'verikasi':'verifikasi','verivikasi':'verifikasi', "veripikasi":"verifikasi",
    "verivikasi":"verifikasi",'verfikasi':'verifikasi','wakanda': 'indonesia',
    'webnya':'web-nya','wkwkwk':'tertawa','yg':'yang', 'kecillllllll':'kecil',
    'tertaiiiiiiiiii':'tai', 'elegal':'ilegal', 'gw' : 'saya',
    'ga': 'tidak', 'gak': 'tidak', 'gk': 'tidak', 'gajelasss' :'jelek',
    'sok²an' : 'berlagak', 'gitu': 'begitu', 'blank':'blank','stuck':'stuck',
    'guguk': 'anjing', 'ngeribetin': 'ribet', 'time out':'timeout',
    'gimana': 'bagaimana', 'liat': 'lihat', 'loadingnya':'loading',
    'dgn': 'dengan', 'melulu': 'terus', 'cape': 'lelah', 'ngehang':'hang',
    'cewe': 'perempuan', "otpnya":"otp",'otep':'otp', 'otpnya':'otp', 'kodeotp':'otp',
    'okay': 'ok', 'tapi': 'tetapi','yaww': 'ya','kayak':'seperti','opo': 'apa',
    'udah': 'sudah', 'memgalihkan': 'mengalihkan', 'dah': 'deh',
    'berkali-kali': 'beberapa kali', 'nyoba': 'coba',
    'ngurus': 'urus', 'pinter': 'pintar', 'gua': 'saya',
    'ngakalin': 'mengakali', 'kayaknya': 'sepertinya', 'ni': 'ini',
    'berfaedah': 'bermanfaat', 'tau': 'tahu', 'ngirim': 'mengirim',
    'ngeberantas': 'memberantas', 'kentirrs': 'gila',
    'dipake': 'dipakai', 'kaya': 'seperti', 'gini': 'begini','apaan': 'apa',
    'sperti': 'seperti','usahh': 'perlu','boss': 'bos', 'upss': 'ups',
    'gausah':'tidak perlu', 'polri': 'polri','aelah': 'lah', 'ttep': 'tetap',
    'uda': 'udah', 'gabisa': 'tidak bisa','ngelanjutinnnn': 'melanjutkan',
    'lg': 'lagi', 'refres': 'refresh','beberapaa': 'beberapa',
    'bermànpaat': 'bermanfaat', 'fiks': 'fix','disemua': 'di semua',
    'nonor': 'nomor', 'n': 'dan', 'lgi': 'lagi',
    'bener': 'benar', 'bs': 'bisa','klo': 'kalau', 'd': 'di', 'cuma': 'hanya',
    'dech': 'deh', 'nich': 'nih', 'mnta': 'minta','tp': 'tetapi','tdk':'tidak',
    'peru': 'perlu','sippp': 'sip', 'pasw': 'password', 'tetep': 'tetap',
    'disalahin': 'disalahkan', 'didownload': 'download', 'afk': 'aplikasi',
    'flay': 'play', 'epin': 'efin', 'ny': 'nya','halah':'malah','jgn':'jangan',
    'poll': 'banget','dsni': 'disini', 'userid': 'user id', 'ps': 'password',
    'pake': 'pakai', 'ttp': 'tetap', 'sdh': 'sudah','entah': 'tidak tahu',
    'gercep': 'gerak cepat','koq': 'kok', 'min': 'admin',
    'kaga': 'tidak','mulu': 'terus', 'amat': 'banget', 'retting':'rating',
    'ttd':'tanda tangan','aj' : 'saja', 'ae' : 'saja', 'aktivasi' : 'aktivasi',
    'aktifasi' : 'aktivasi','aktivin' : 'aktifkan', 'aktivis' : 'aktivasi',
    'aktivasiii' : 'aktivasi', 'aktik' : 'aktif','amat' : 'banget',
    'amatt' : 'banget', 'amaaaat' : 'banget', 'amang"' : 'paman',
    'ampuuuuuuuunnnnnn' : 'ampun', 'ancene' : 'memang', 'angel' : 'sulit',
    'apl' : 'aplikasi', 'apliksi' : 'aplikasi', 'aflikasi' : 'aplikasi',
    'apkikasi' : 'aplikasi', 'aturen' : 'atur', 'asuuuuuuuu' : 'anjing',
    'bafring' : 'buffering','bangettt' : 'banget', 'bangetttt' : 'banget',
    'bangetttttttttt' : 'banget', 'bat' : 'banget','bbeberapa' : 'beberapa',
    'becus' : 'mampu', 'becuss' : 'becus', 'beganda' : 'ganda',
    'bejatt' : 'bejat','belom' :'belum','berahir':'berakhir','beresss':'selesai',
    'bet' : 'banget', 'betol2' : 'betul', 'bgtt' : 'banget', 'bgttt' : 'banget',
    'bguss' : 'bagus', 'bgt' : 'banget', 'bikan' : 'bikin', 'bintg' : 'bintang',
    'blass' : 'sama sekali', 'blom' : 'belum', 'bodok' : 'bodoh', 'bnerrr' : 'benar',
    'boro²' : 'jangankan', 'bosss' : 'bos', 'bott' : 'bot', 'btl' : 'betul',
    'buanh' : 'buang', 'bundeli' : 'rumit', 'bus00x' : 'busuk', 'busuukkk' : 'busuk',
    'buuusuuk' : 'busuk','captca' : 'captcha', "busuuuk":"busuk","busuuukk":"busuk",
    "busukkk":"busuk",'cape' : 'capek', 'cipatin' : 'menciptakan', 'cm' : 'hanya',
    'cmn' : 'hanya','coke' : 'cuk', 'cooegg' : 'cuk', 'cortax' : 'coretax',
    'cortec' : 'coretax','coretak' : 'coretax', 'koretak' : 'coretax',
    'kortek' :'coretax', 'dbuat' : 'dibuat','didlm' : 'dalam',
    'digedein' : 'dibesarkan', 'dimna' : 'dimana', 'djancokkk' : 'jancuk',
    'djponline' : 'djp online', 'dobol' : 'bodoh', 'dongo' : 'dungu', 'doang' : 'hanya',
    'downloan' : 'download', 'dpke' : 'dipakai', 'eek' : 'tai', 'feling' : 'filing',
    'ferivikasi' : 'verifikasi', 'gaguna' : 'tidak berguna', 'gagunaaaaaaaaaaa' : 'tidak guna',
    'gae' : 'buat', 'gaes' : 'guys', 'gaje' : 'tidak jelas', 'gasih' : 'tidak',
    'gedee' : 'besar', 'gedeg' : 'kesal', 'gehh' : 'kok', 'ge' : 'tuh',
    'gerecep' : 'gerak cepat', 'geteg' : 'kesal', 'gj' : 'tidak jelas', 'gjls' : 'tidak jelas',
    'gimanamau' : 'bagaimana', 'goblokk' : 'goblok', 'goood' : 'bagus', 'gx' : 'tidak',
    'hadeuh' : 'hadeh', 'hadeeeh' : 'aduh', 'hadehhh' : 'hedeh', 'hamburakan' : 'hamburkan',
    'harraaammmmm' : 'haram', 'herman' : 'heran', 'hmm' : 'hm', 'idup' : 'hidup',
    'ihhh' : 'ih', 'iki' : 'ini', 'indifikasi' : 'indikasi', 'jadul' : 'kuno',
    'bahela' : 'kuno', 'jalok' : 'minta', 'jane' : 'sebenarnya', 'jelleekkk' : 'jelek',
    'jelekkkkkkk' : 'jelek', 'jemboet' : 'jembut', 'jga' : 'juga', 'jink' : 'anjing',
    'jir' : 'anjing', 'jlk' : 'jelek', 'jncox' : 'jancuk', 'kaga' : 'tidak',
    'kaggk' : 'tidak', 'kasi' : 'kasih', 'kayak' : 'seperti', 'keat' : 'tai',
    'kek' : 'seperti', 'keluatan' : 'keluaran', 'kg' : 'tidak', 'kl' : 'kalau',
    'klok' : 'kalau', 'klw' : 'kalau', 'knp' : 'kenapa', 'ko' : 'kok',
    'koq' : 'kok', 'kon62829' : 'kontol', 'kont' : 'kontol', 'konto1' : 'kontol',
    'kontl' : 'kontol', 'kontoll' : 'kontol', 'kontil' : 'kontol', 'kntlll' : 'kontol',
    'konxol' : 'kontol', 'kono' : 'sana', 'koropsi' : 'korupsi', 'kudet' : 'kurang update',
    'kudu' : 'harus', 'kuatir' : 'khawatir', 'kuedit' : 'edit', 'kyk' : 'seperti',
    'lalot' : 'lemot', 'lemootttttt' : 'lemot',
    'lemottttt' : 'lemot', 'lgsg' : 'langsung', 'log in' : 'login', 'loe thong' : 'jancuk',
    'lu' : 'kamu', 'lupah' : 'lupa', 'lw' : 'kalau', 'make' : 'pakai',
    'manjang' : 'lama', 'masok' : 'masuk', 'masuklaman' : 'masuk laman', 'mboh' : 'tidak',
    'mgkn' : 'mungkin', 'mnuntut' : 'menuntut', 'mnaa' : 'mana', 'modyarrr' : 'mati',
    'mpajka' : 'mpajak', 'mpwp' : 'npwp', 'mruni' : 'murni', 'msh' : 'masih',
    'muter²' : 'loading', 'mw' : 'mau', 'naajissss' : 'najis', 'nda' : 'tidak',
    'ndk' : 'tidak', 'ngabikin' : 'buat', 'ngabisin' : 'menghabiskan', 'ngaco' : 'mengacau',
    'ngehek' : 'hack', 'ngak' : 'tidak', 'niceee' : 'nice', 'nik+kk' : 'nik kk',
    'nik/npwp' : 'nik dan npwp', 'nomot' : 'nomor', 'nope' : 'tidak', 'nopo' : 'apa',
    'nyesel' : 'menyesal', 'nyieun' : 'membuat', 'nyoba' : 'mencoba', 'nyusain' : 'menyusahkan',
    'okee' : 'oke', 'olok' : 'tukar', 'onlen' : 'online', 'oplas' : 'operasi plastik',
    'opo' : 'apa', 'orng' : 'orang', 'pajakk' : 'pajak', 'pantekk' : 'sialan',
    'paok' : 'bodoh', 'passphares' : 'passphrase', 'pasword' : 'password', 'pekok' : 'dungu',
    'periv' : 'verifikasi', 'petbaiki' : 'perbaiki', 'pisann' : 'banget', 'pisan' : 'sekali',
    'plenger' : 'konyol', 'plis' : 'please', 'plsa' : 'pulsa', 'poto' : 'foto',
    'puyeng' : 'pusing', 'pw' : 'password', 'qontol' : 'kontol', 'qontollll' : 'kontol',
    'ra' : 'tidak', 'rada' : 'agak', 'rebu' : 'ribu', 'regis' : 'registrasi',
    'renting' : 'rating', 'ribeettt' : 'rumit', 'ribetttttt' : 'rumit', 'rudet' : 'susah',
    'ruwet' : 'rumit', 'sabgat' : 'sangat', 'sampe' : 'sampai', 'sampahhh' : 'sampah',
    'sampaaahhh' : 'sampah', 'sekalas' : 'sekelas', 'sejam' : 'satu jam', 'sihh' : 'sih',
    'siaa' : 'sia', 'sistrm' : 'sistem', 'sl' : 'selalu', 'slalu' : 'selalu',
    'sllu' : 'selalu', 'sm' : 'sama', 'smakin' : 'semakin', 'smua' : 'semua',
    'so soan' : 'berlagak', 'sperti' : 'seperti', 'stresss' : 'stres', 'stuk' : 'stuck',
    'suda' : 'sudah', 'sulitt' : 'sulit', 'suasu' : 'anjing', 'suuulit' : 'sulit',
    'swabfoto' : 'swafoto', 'swap' : 'swafoto', 'swapfoto' : 'swafoto', 'syudahlah' : 'sudahlah',
    't4ik' : 'tai', 'ta*k' : 'tai', 'taik' : 'tai', 'taii' : 'tai', "jelekk":"jelek","jelekkk":"jelek",
    'taik' : 'tai', 'temen' : 'teman', 'terdikasi' : 'terintegrasi', 'terkontol' : 'kontol',
    'terkontoI' : 'kontol', 'tidk' : 'tidak', 'tilep' : 'mencuri', 'tlfn' : 'telepon',
    'tlong' : 'tolong', 'tlpn' : 'telepon', 'tnggu' : 'tunggu', 'to long' : 'tolong',
    'toll' : 'tolol', 'tol*ol' : 'tolol', 'tolongggggg' : 'tolong', 'tpi' : 'tapi',
    'tpii' : 'tapi', 'trs' : 'terus', 'ttd' : 'tanda tangan', 'tuhhh' : 'tuh',
    'uweee' : 'lama', 'ud' : 'udah', 'udh' : 'sudah', 'usahh' : 'perlu',
    'vermuk' : 'verifikasi muka', 'vervikasi' : 'verifikasi', "paraah":"parah",
    'wab' : 'website', 'webite' : 'website', 'weleh' : 'geleng', 'wei' : 'hei',
    'wes' : 'sudah', 'woei' : 'woi', 'wong' : 'orang', 'wp' : 'wajib pajak',
    'wnwp' : 'npwp', 'x' : 'kali', 'y' : 'ya', 'yeeay' : 'yes', "ampaas":"ampas",
    'yok' : 'yuk', 'yutub' : 'yotube', 'yy' : 'ya', 'teros' : 'terus',"tolool":"tolol",
    "tololl":"tolol","okee":"oke","makasihh":"terima kasih",'veripikasi':'verifikasi',
    'verikasi':'verifikasi', 'periv':'verifikasi', 'ngelogin':'login',
    'garespon':'tidak respon', 'garespons':'tidak respon', 'gkmasuk':'tidak masuk',
    'gabukak':'tidak buka', 'gabuka':'tidak buka', 'gklogin':'tidak login',
    'gamasuk':'tidak masuk','gamasukin':'tidak masuk', 'gkbisalogin':'tidak bisa login',
    'gabisalogin':'tidak bisa login', 'ngespam':'spam', 'keblok':'blokir',
    'keblokir':'blokir', 'diblok':'blokir', 'diblokir':'blokir', 'dilaptop':'di laptop','qntl':'kontol','parahhh':'parah'
    }

# --- FUNGSI PREPROCESSING LENGKAP ---
def cleaning_raw(text):
    text = str(text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = text.replace('.', ' ').replace('_', ' ')
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def remove_repeated_characters(text):
    # Mengubah karakter yang berulang lebih dari 2 kali menjadi 2 (contoh: "paraaah" -> "paraah")
    # Kemudian dilakukan stemming agar "paraah" -> "parah"
    return re.sub(r'(.)\1{2,}', r'\1', text)

def get_full_pipeline(text):
    # 1. Cleaning
    s1 = cleaning_raw(text)
    # 2. Case Folding
    s2 = s1.lower()
    # 3. Handle Repeated Characters (Baru)
    s3_raw = remove_repeated_characters(s2)
    # 4. Tokenization
    s3 = s3_raw.split()
    # 4. Slang Normalization
    s4 = [slang_dict.get(w, w) for w in s3]
    # 5. Stopword Removal
    s5 = [w for w in s4 if w not in stop_words]
    # 6. Stemming
    s6 = [stemmer.stem(w) for w in s5]
    
    final_text = ' '.join(s6)
    return s1, s2, s3, s4, s5, s6, final_text

# --- LOAD MODEL ---
@st.cache_resource
def load_assets():
    model = joblib.load('svm_calibrated_model.pkl')
    tfidf = joblib.load('tfidf.pkl')
    return model, tfidf

model, tfidf = load_assets()

# --- UI STREAMLIT ---
# Layout Judul & Logo
col1, col2 = st.columns([1, 8])
with col1:
    st.image("logo.png", width=70) 
with col2:
    st.title("Analisis Sentimen M-Pajak")

# Sidebar
with st.sidebar:
    st.header("📌 Informasi Model")
    st.info("Analisis sentimen real-time untuk ulasan M-Pajak.")
    
    st.markdown("- **Algoritma:** LinearSVC (Calibrated)")
    st.markdown("- **Feature Extraction:** TF-IDF Vectorizer")
    st.markdown("- **Dataset:** 5,500 Ulasan Terbaru (Google Play Store)")
    st.markdown("- **Akurasi:** 81.03%")
    
    with st.expander("💡 Mengapa Model Ini?"):
        st.write("""
        *   **TF-IDF:** Optimal untuk pembobotan kata penting dan peredaman *noise*.
        *   **LinearSVC:** Efektif untuk teks berdimensi tinggi, menyeimbangkan **kecepatan prediksi** dan **akurasi**.
        *   **Calibration:** Memberikan *Probability Score* untuk transparansi tingkat keyakinan model.
        
        *Solusi ini dipilih karena ringan dan optimal untuk kebutuhan real-time.*
        """)
        
st.markdown("---") # Garis pembatas
st.markdown("### 👨‍💻 Developed by:")
st.caption("Hasti Sri Fatmawati")        

# Main Area
st.write("Dashboard ini memprediksi apakah ulasan pengguna bersifat **Positif, Negatif, atau Netral**.")
user_input = st.text_area("Tulis ulasan aplikasi di sini:", placeholder="Contoh: Login lemot banget, apk ga bisa dibuka")

if st.button("Analisis Sentimen"):
    if user_input:
        # Jalankan pipeline
        s1, s2, s3, s4, s5, s6, final_txt = get_full_pipeline(user_input)
        vec = tfidf.transform([final_txt])
        
        # Prediksi
        prediction = model.predict(vec)[0]
        confidence = model.predict_proba(vec).max() * 100
        
        # Display Result
        c1, c2 = st.columns(2)
        c1.metric("Hasil Sentimen", prediction.upper())
        c2.metric("Confidence", f"{confidence:.2f}%")
        
        # Transparansi Pipeline
        with st.expander("🔍 Lihat Detail Pipeline Preprocessing"):
            st.write(f"**1. Cleaning:** `{s1}`")
            st.write(f"**2. Case Folding:** `{s2}`")
            st.write(f"**3. Tokenization:** `{s3}`")
            st.write(f"**4. Slang Normalization:** `{s4}`")
            st.write(f"**5. Stopword Removal:** `{s5}`")
            st.write(f"**6. Stemming:** `{s6}`")
            st.success(f"Final Input: `{final_txt}`")
    else:
        st.warning("Silakan masukkan teks ulasan terlebih dahulu!")
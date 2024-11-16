import qrcode
from qrcode.constants import ERROR_CORRECT_L

URL = 'https://github.com/AlphaGergedan/Sampling-HNNs'
SAVE_DIR = './assets/qr-code.png'

qr = qrcode.QRCode(version=1, error_correction=ERROR_CORRECT_L, box_size=10, border=4)
qr.add_data(URL)
qr.make(fit=True)

img = qr.make_image(fill_color='black', back_color='white')
img.save(SAVE_DIR)

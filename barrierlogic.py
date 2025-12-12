import serial
import time

# ========================================
# KONFIGURASI - GANTI SESUAI PORT KAMU
# ========================================
SERIAL_PORT = 'COM3'  # Windows: COM3, COM4, dll | Linux: /dev/ttyUSB0
BAUD_RATE = 9600

# ========================================
# INISIALISASI KONEKSI
# ========================================
try:
    relay = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    print("✅ Terhubung ke relay module")
    time.sleep(1)
except Exception as e:
    print(f"❌ Error: {e}")
    exit()

# ========================================
# FUNGSI KONTROL RELAY
# ========================================
def buka_barrier():
    """Trigger barrier OPEN (relay ON 500ms)"""
    print("🟢 MEMBUKA BARRIER...")
    relay.setDTR(True)   # Relay ON
    time.sleep(0.5)      # Hold 500ms
    relay.setDTR(False)  # Relay OFF
    print("✅ Barrier terbuka!\n")

def tutup_barrier():
    """Trigger barrier CLOSE (relay OFF atau command CLOSE)"""
    print("🔴 MENUTUP BARRIER...")
    relay.setDTR(False)  # Relay OFF (barrier auto-close atau manual)
    print("✅ Barrier tertutup!\n")

# ========================================
# PROGRAM UTAMA
# ========================================
print("="*40)
print("🚧 KONTROL BARRIER GATE FAAC 615")
print("="*40)
print("Ketik 1 untuk BUKA")
print("Ketik 0 untuk TUTUP")
print("Ketik 'q' untuk KELUAR")
print("="*40 + "\n")

try:
    while True:
        # Input user
        perintah = input("Masukkan perintah (1/0/q): ").strip()
        
        if perintah == '1':
            buka_barrier()
            
        elif perintah == '0':
            tutup_barrier()
            
        elif perintah.lower() == 'q':
            print("\n👋 Program selesai")
            break
            
        else:
            print("❌ Input salah! Ketik 1, 0, atau q\n")
            
except KeyboardInterrupt:
    print("\n\n⚠️ Program dihentikan")
    
finally:
    relay.close()
    print("Koneksi ditutup")

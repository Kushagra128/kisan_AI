"""
Patches Windows registry so that modern OneCore TTS voices
(added via Settings > Speech) become visible to SAPI5 / pyttsx3.
Run once as Administrator.
"""
import winreg

ONECORE_PATH = r"SOFTWARE\Microsoft\Speech_OneCore\Voices\Tokens"
SAPI5_PATH   = r"SOFTWARE\Microsoft\Speech\Voices\Tokens"

def copy_voices():
    added = 0
    try:
        onecore = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, ONECORE_PATH)
    except FileNotFoundError:
        print("OneCore voices key not found – nothing to copy.")
        return

    i = 0
    while True:
        try:
            name = winreg.EnumKey(onecore, i)
            i += 1
        except OSError:
            break

        src_path  = f"{ONECORE_PATH}\\{name}"
        dest_path = f"{SAPI5_PATH}\\{name}"

        try:
            # Check if already exists in SAPI5
            winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, dest_path)
            print(f"  [SKIP] {name} (already exists)")
            continue
        except FileNotFoundError:
            pass

        # Copy the key and its values
        try:
            src_key  = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, src_path)
            dest_key = winreg.CreateKey(winreg.HKEY_LOCAL_MACHINE, dest_path)

            j = 0
            while True:
                try:
                    val_name, val_data, val_type = winreg.EnumValue(src_key, j)
                    winreg.SetValueEx(dest_key, val_name, 0, val_type, val_data)
                    j += 1
                except OSError:
                    break

            # Copy attributes sub-key if it exists
            try:
                attr_src  = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, src_path + "\\Attributes")
                attr_dest = winreg.CreateKey(winreg.HKEY_LOCAL_MACHINE, dest_path + "\\Attributes")
                k = 0
                while True:
                    try:
                        val_name, val_data, val_type = winreg.EnumValue(attr_src, k)
                        winreg.SetValueEx(attr_dest, val_name, 0, val_type, val_data)
                        k += 1
                    except OSError:
                        break
            except FileNotFoundError:
                pass

            print(f"  [ADDED] {name}")
            added += 1
        except Exception as e:
            print(f"  [ERROR] {name}: {e}")

    print(f"\nDone! {added} voice(s) added to SAPI5.")
    if added:
        print("Please RESTART your Django server now.")

if __name__ == "__main__":
    copy_voices()

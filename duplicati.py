import os

def trova_differenze(dir_immagini, dir_testi):
    try:
        # Estrae i nomi base (senza estensione) dei file presenti nelle cartelle
        immagini = {os.path.splitext(f)[0] for f in os.listdir(dir_immagini) if os.path.isfile(os.path.join(dir_immagini, f))}
        testi = {os.path.splitext(f)[0] for f in os.listdir(dir_testi) if os.path.isfile(os.path.join(dir_testi, f))}
    except FileNotFoundError as e:
        print(f"Errore: Una delle cartelle specificate non esiste. Dettagli: {e}")
        return

    # Trova le discrepanze usando i set
    solo_immagini = immagini - testi
    solo_testi = testi - immagini

    print("--- RISULTATI DEL CONFRONTO ---")
    if not solo_immagini and not solo_testi:
        print("Perfetto! Tutti i file di immagine hanno un file di testo corrispondente e viceversa.")
    else:
        if solo_immagini:
            print(f"\nTrovate {len(solo_immagini)} immagini SENZA il file di testo corrispondente:")
            for nome in sorted(solo_immagini):
                print(f"  - {nome}")
        
        if solo_testi:
            print(f"\nTrovati {len(solo_testi)} file di testo SENZA l'immagine corrispondente:")
            for nome in sorted(solo_testi):
                print(f"  - {nome}")

if __name__ == "__main__":
    # Modifica questi percorsi inserendo i percorsi reali delle tue cartelle
    cartella_immagini = r".\dataset\images\validation"
    cartella_testi =    r".\dataset\labels\validation"
    
    trova_differenze(cartella_immagini, cartella_testi)
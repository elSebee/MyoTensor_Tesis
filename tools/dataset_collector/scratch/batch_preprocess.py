"""
=============================================================
 batch_preprocess.py — Preprocesamiento masivo del dataset
=============================================================
 Aplica de forma automática el nuevo preprocess.py en todos
 los archivos session_*.csv de todos los sujetos (S02 - S15).
 Actualiza cada archivo in-place agregando:
   - emg_norm (normalizada por MVC)
   - restimulus (onset y offset dinámicos)
   - valid_flag (detección robusta de outliers)
=============================================================
"""

import sys
import pandas as pd
from pathlib import Path

# Agregar la ruta actual al path para asegurar importes relativos correctos
sys.path.append(str(Path(__file__).parent))
from preprocess import run_preprocess

def main():
    base_dir = Path("/home/cbe/Proyectos/MyoTensor_Tesis/intelligence/datasets/raw/myotensor_proto")
    if not base_dir.exists():
        print(f"[ERROR] El directorio base {base_dir} no existe.")
        sys.exit(1)
        
    print(f"[*] Iniciando preprocesamiento masivo sEMG en: {base_dir}\n")
    
    # Encontrar todas las carpetas de sujetos (v.g. S02, S07, etc.)
    subject_dirs = sorted([d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("S")])
    
    total_processed = 0
    errors = 0
    
    print(f"[+] Se encontraron {len(subject_dirs)} carpetas de sujetos: {[d.name for d in subject_dirs]}\n")
    print(f"{'Sujeto':<8} | {'Archivo CSV de Sesión':<30} | {'Muestras':<10} | {'Activas':<8} | {'Outliers':<10}")
    print("-" * 76)
    
    for s_dir in subject_dirs:
        # Encontrar archivos CSV
        csv_files = sorted(list(s_dir.glob("session_*.csv")))
        
        for csv_path in csv_files:
            # Ignorar backups o temporales
            if "backup" in csv_path.name or "test" in csv_path.name:
                continue
                
            # Encontrar el archivo de metadata JSON correspondiente
            meta_name = csv_path.name.replace(".csv", "_metadata.json")
            meta_path = s_dir / meta_name
            
            if not meta_path.exists():
                print(f"[WARN] No se encontró metadata para {csv_path.name}, se omite.")
                continue
                
            try:
                # Ejecutar el preprocesador modular
                run_preprocess(str(csv_path), str(meta_path))
                
                # Cargar estadísticas de validación
                df = pd.read_csv(csv_path)
                total_samples = len(df)
                active_samples = (df['restimulus'] != 0).sum() if 'restimulus' in df.columns else 0
                invalid_samples = (df['valid_flag'] == 0).sum() if 'valid_flag' in df.columns else 0
                
                print(f"{s_dir.name:<8} | {csv_path.name:<30} | {total_samples:<10} | {active_samples:<8} | {invalid_samples:<10}")
                total_processed += 1
            except Exception as e:
                print(f"[ERROR] Falló procesamiento en {csv_path.name}: {e}")
                errors += 1
                
    print("\n" + "=" * 76)
    print(f"[+] ¡Preprocesamiento Masivo Terminado!")
    print(f"    - Sesiones procesadas con éxito: {total_processed}")
    print(f"    - Errores encontrados:           {errors}")
    print("=" * 76)

if __name__ == "__main__":
    main()

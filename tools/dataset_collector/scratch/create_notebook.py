import json
import os

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Preprocesamiento, Ventaneo y División del Dataset para MyoTensor Proto (S07)\n",
    "\n",
    "Este notebook realiza el pipeline completo de procesamiento para el canal único de sEMG de **MyoTensor Proto** (S07):\n",
    "1. Filtro de outliers (`valid_flag == 1`).\n",
    "2. Segmentación contigua por bloques de estado de estímulo (evita mezclar tiempos no contiguos).\n",
    "3. Extracción de ventanas deslizantes con solapamiento ($W=300$ ms, $S=100$ ms).\n",
    "4. División en conjuntos de **Entrenamiento (80%)** y **Prueba (20%)** estratificados.\n",
    "5. Exportación de tensores en formato NumPy binario (`.npy`), incluyendo etiquetas categóricas (para SVM/RF) y etiquetas one-hot (para tu red neuronal CNN-LSTM).\n",
    "\n",
    "---"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## ¿Qué Estructura Tienen los Tensores Finales?\n",
    "\n",
    "### Conjunto de Entrenamiento (80%):\n",
    "*   **`X_train.npy`:** Tensor 3D de shape `(1638, 300, 1)` (1,638 ventanas de 300 ms con 1 canal).\n",
    "*   **`y_train.npy`:** Vector 1D de shape `(1638,)` con etiquetas enteras (`0, 1, 2, 3`) para SVM/Random Forest.\n",
    "*   **`y_train_onehot.npy`:** Matriz 2D de shape `(1638, 4)` con codificación one-hot para tu red **CNN-LSTM**.\n",
    "\n",
    "### Conjunto de Prueba/Validación (20%):\n",
    "*   **`X_test.npy`:** Tensor 3D de shape `(410, 300, 1)` (410 ventanas de 300 ms con 1 canal).\n",
    "*   **`y_test.npy`:** Vector 1D de shape `(410,)` con etiquetas enteras (`0, 1, 2, 3`) para SVM/Random Forest.\n",
    "*   **`y_test_onehot.npy`:** Matriz 2D de shape `(410, 4)` con codificación one-hot para tu red **CNN-LSTM**."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Configuración de Parámetros Globales"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "from pathlib import Path\n",
    "from sklearn.model_selection import train_test_split\n",
    "from tensorflow.keras.utils import to_categorical\n",
    "\n",
    "# ======================================================================\n",
    "# PARÁMETROS CONFIGURABLES\n",
    "# ======================================================================\n",
    "W = 300         # Tamaño de la ventana (300 ms a 1000 Hz)\n",
    "S = 100         # Paso de la ventana (100 ms para 66.6% de solapamiento)\n",
    "TEST_SIZE = 0.2 # 20% para el conjunto de prueba (Test/Validation)\n",
    "SEED = 42       # Semilla para que el split sea reproducible\n",
    "BASE_DIR = Path(\"../../datasets/raw/myotensor_proto\")\n",
    "OUTPUT_DIR = Path(\"../../datasets/processed/myotensor_proto\")\n",
    "OUTPUT_DIR.mkdir(parents=True, exist_ok=True)\n",
    "# ======================================================================"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Algoritmo de Preprocesamiento y Extracción de Ventanas"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "X_list = []\n",
    "y_list = []\n",
    "\n",
    "# Buscar específicamente la sesión CSV de S07\n",
    "csv_files = sorted(list(BASE_DIR.glob(\"S07/session_*.csv\")))\n",
    "print(f\"[*] PASO 1: Buscando archivos CSV en {BASE_DIR}\")\n",
    "print(f\"    -> Se encontró el archivo de S07: {[f.name for f in csv_files]}\\n\")\n",
    "\n",
    "for csv_path in csv_files:\n",
    "    print(f\"[*] PASO 2: Cargando archivo {csv_path.name}...\")\n",
    "    df = pd.read_csv(csv_path)\n",
    "    print(f\"    -> Muestras totales en crudo: {len(df)}\")\n",
    "    \n",
    "    # 1. Filtro de muestras de calidad valid_flag == 1\n",
    "    df_valid = df[df[\"valid_flag\"] == 1].copy()\n",
    "    n_outliers = len(df) - len(df_valid)\n",
    "    print(f\"[*] PASO 3: Aplicando filtro de calidad (valid_flag == 1)\")\n",
    "    print(f\"    -> Muestras válidas conservadas: {len(df_valid)} (outliers eliminados: {n_outliers})\")\n",
    "    \n",
    "    # 2. Identificar bloques contiguos de restimulus (gestos o reposos)\n",
    "    block_id = (df_valid[\"restimulus\"] != df_valid[\"restimulus\"].shift()).cumsum()\n",
    "    groups = df_valid.groupby(block_id)\n",
    "    \n",
    "    print(f\"[*] PASO 4: Segmentando señal en bloques contiguos de actividad/reposo...\")\n",
    "    print(f\"    -> Se detectaron {len(groups)} bloques contiguos de señal en la sesión.\\n\")\n",
    "    print(f\"{'Bloque ID':<10} | {'Clase Gesto':<12} | {'Duración (muestras)':<20} | {'Ventanas Extraídas':<20}\")\n",
    "    print(\"-\" * 70)\n",
    "    \n",
    "    block_count = 1\n",
    "    gesture_names = {0: \"Reposo (0)\", 1: \"Palma (1)\", 2: \"Puño (2)\", 3: \"Paz (3)\"}\n",
    "    \n",
    "    for g_id, group_df in groups:\n",
    "        gesture_class = group_df[\"restimulus\"].iloc[0]\n",
    "        class_name = gesture_names.get(gesture_class, f\"Clase {gesture_class}\")\n",
    "        \n",
    "        # Señal normalizada\n",
    "        sig_col = \"emg_norm\" if \"emg_norm\" in group_df.columns else \"filtered\"\n",
    "        signal_block = group_df[sig_col].values\n",
    "        L = len(signal_block)\n",
    "        \n",
    "        # Si el bloque es más corto que la ventana configurable W, se descarta\n",
    "        if L < W:\n",
    "            print(f\"Block {block_count:<5} | {class_name:<12} | {L:<20} | 0 (Descartado: L < W)\")\n",
    "            block_count += 1\n",
    "            continue\n",
    "            \n",
    "        # 3. Ventaneo Deslizante\n",
    "        extracted_in_block = 0\n",
    "        for start in range(0, L - W + 1, S):\n",
    "            window = signal_block[start:start + W]\n",
    "            X_list.append(window)\n",
    "            y_list.append(gesture_class)\n",
    "            extracted_in_block += 1\n",
    "            \n",
    "        print(f\"Block {block_count:<5} | {class_name:<12} | {L:<20} | {extracted_in_block:<20}\")\n",
    "        block_count += 1\n",
    "\n",
    "print(f\"\\n[+] PASO 5: Ventaneo finalizado. Total de ventanas generadas: {len(X_list)}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. División del Dataset (Train/Test Split) con Estratificación"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(\"[*] PASO 6: Convirtiendo listas a arreglos NumPy...\")\n",
    "X = np.array(X_list)\n",
    "y = np.array(y_list)\n",
    "\n",
    "# Expandir dimensión de canal único sEMG: (num_ventanas, W, 1)\n",
    "X = np.expand_dims(X, axis=-1)\n",
    "y = y.astype(np.int32)\n",
    "\n",
    "print(f\"[*] PASO 7: Dividiendo en conjuntos de Entrenamiento y Prueba ({int((1-TEST_SIZE)*100)}% / {int(TEST_SIZE*100)}%)...\")\n",
    "# El parámetro 'stratify=y' es vital para mantener exactamente la misma proporción de gestos en train y test\n",
    "X_train, X_test, y_train, y_test = train_test_split(\n",
    "    X, y, \n",
    "    test_size=TEST_SIZE, \n",
    "    stratify=y, \n",
    "    random_state=SEED\n",
    ")\n",
    "\n",
    "print(f\"    -> Conjunto de Entrenamiento (Shape): X_train={X_train.shape}, y_train={y_train.shape}\")\n",
    "print(f\"    -> Conjunto de Prueba (Shape):        X_test={X_test.shape}, y_test={y_test.shape}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Codificación One-Hot para Redes Neuronales (Keras / TensorFlow)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(\"[*] PASO 8: Creando codificación One-Hot para el pipeline de Redes Neuronales...\")\n",
    "num_classes = len(np.unique(y))\n",
    "y_train_onehot = to_categorical(y_train, num_classes=num_classes)\n",
    "y_test_onehot = to_categorical(y_test, num_classes=num_classes)\n",
    "\n",
    "print(f\"    -> y_train (One-Hot Shape): {y_train_onehot.shape}\")\n",
    "print(f\"    -> y_test (One-Hot Shape):  {y_test_onehot.shape}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. Guardando Tensores en Carpeta de Procesados"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(f\"[*] PASO 9: Guardando todos los tensores binarios en {OUTPUT_DIR}...\")\n",
    "\n",
    "# Guardar datos de entrenamiento\n",
    "np.save(OUTPUT_DIR / 'X_train.npy', X_train)\n",
    "np.save(OUTPUT_DIR / 'y_train.npy', y_train)\n",
    "np.save(OUTPUT_DIR / 'y_train_onehot.npy', y_train_onehot)\n",
    "\n",
    "# Guardar datos de prueba/validación\n",
    "np.save(OUTPUT_DIR / 'X_test.npy', X_test)\n",
    "np.save(OUTPUT_DIR / 'y_test.npy', y_test)\n",
    "np.save(OUTPUT_DIR / 'y_test_onehot.npy', y_test_onehot)\n",
    "\n",
    "print(\"\\n[OK] ¡Guardado Completado con Éxito!\")\n",
    "print(f\"     -> X_train.npy        : {X_train.shape}\")\n",
    "print(f\"     -> y_train.npy        : {y_train.shape} (Categorías)\")\n",
    "print(f\"     -> y_train_onehot.npy : {y_train_onehot.shape} (One-Hot)\")\n",
    "print(f\"     -> X_test.npy         : {X_test.shape}\")\n",
    "print(f\"     -> y_test.npy         : {y_test.shape} (Categorías)\")\n",
    "print(f\"     -> y_test_onehot.npy  : {y_test_onehot.shape} (One-Hot)\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 6. Verificación de Balance y Distribución de Clases en los Splits"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(\"[*] PASO 10: Verificando estratificación y balance de clases...\\n\")\n",
    "\n",
    "train_classes, train_counts = np.unique(y_train, return_counts=True)\n",
    "test_classes, test_counts = np.unique(y_test, return_counts=True)\n",
    "gesture_names = {0: \"Reposo (0)\", 1: \"Palma (1)\", 2: \"Puño (2)\", 3: \"Paz (3)\"}\n",
    "\n",
    "print(\"Distribución de clases en Conjunto de Entrenamiento (Train Split):\")\n",
    "print(\"=\" * 65)\n",
    "for c, cnt in zip(train_classes, train_counts):\n",
    "    name = gesture_names.get(c, f\"Clase {c}\")\n",
    "    pct = (cnt / len(y_train)) * 100\n",
    "    print(f\"  * {name:<12} : {cnt:<5} ventanas ({pct:.2f}% del split)\")\n",
    "print(\"=\" * 65)\n",
    "\n",
    "print(\"\\nDistribución de clases en Conjunto de Prueba (Test Split):\")\n",
    "print(\"=\" * 65)\n",
    "for c, cnt in zip(test_classes, test_counts):\n",
    "    name = gesture_names.get(c, f\"Clase {c}\")\n",
    "    pct = (cnt / len(y_test)) * 100\n",
    "    print(f\"  * {name:<12} : {cnt:<5} ventanas ({pct:.2f}% del split)\")\n",
    "print(\"=\" * 65)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}

output_path = "/home/cbe/Proyectos/MyoTensor_Tesis/intelligence/notebooks/myotensor_proto/procesar_myotensor_proto.ipynb"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print(f"Jupyter Notebook successfully created at: {output_path}")

import os
import pandas as pd

def cargar_datos():
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
    ruta_local = "datos/airline-passengers.csv"
    
    # Intentamos descargar desde internet
    try:
        df = pd.read_csv(url)
    except Exception:
        # Si no hay internet o falla, leemos de la ruta local
        if os.path.exists(ruta_local):
            df = pd.read_csv(ruta_local)
        else:
            raise FileNotFoundError(
                "No se pudo descargar de internet ni se encontró el archivo local en 'datos/airline-passengers.csv'."
            )
            
    # Asegurarnos de que la columna de mes sea de tipo fecha o texto ordenado
    # El CSV trae 'Month' (ej: '1949-01') y 'Passengers' (número)
    return df

if __name__ == "__main__":
    # Prueba rápida del archivo
    df = cargar_datos()
    print("¡Datos cargados con éxito!")
    print(df.head())

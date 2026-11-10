from datos import cargar_datos

def analizar_datos():
    df = cargar_datos()
    
    # Cantidad total de meses (filas)
    total_meses = len(df)
    
    # Desde y hasta cuándo
    mes_inicial = df['Month'].iloc[0]
    mes_final = df['Month'].iloc[-1]
    
    # Mes con menos pasajeros
    min_idx = df['Passengers'].idxmin()
    mes_min = df.loc[min_idx, 'Month']
    pasajeros_min = df.loc[min_idx, 'Passengers']
    
    # Mes con más pasajeros
    max_idx = df['Passengers'].idxmax()
    mes_max = df.loc[max_idx, 'Month']
    pasajeros_max = df.loc[max_idx, 'Passengers']
    
    print("Resumen de los datos de pasajeros de avion:")
    print(f"- Total de meses: {total_meses}")
    print(f"- Desde: {mes_inicial} hasta {mes_final}")
    print(f"- Mes con menos pasajeros: {mes_min} ({pasajeros_min} pasajeros)")
    print(f"- Mes con más pasajeros: {mes_max} ({pasajeros_max} pasajeros)")

if __name__ == "__main__":
    analizar_datos()

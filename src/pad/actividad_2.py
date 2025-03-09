import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

class Ejercicios:
    """
    Clase que encapsula 21 ejercicios con NumPy y Matplotlib,
    y exporta los resultados a un archivo Excel con los gráficos incrustados.
    """

    def __init__(self):
        """
        Constructor: inicializa un DataFrame vacío para almacenar
        los resultados de cada ejercicio.
        """
        self.df = pd.DataFrame(columns=["Punto", "Resultado"])

    # -------------------------------------------------------------------------
    # 1. Generar un array de NumPy con valores desde 10 hasta 29.
    # -------------------------------------------------------------------------
    def punto_1(self):
        array_10_29 = np.arange(10, 30)
        self.df.loc[len(self.df)] = [
            "1. Array de 10 a 29",
            str(array_10_29)
        ]
    
    # -------------------------------------------------------------------------
    # 2. Sumar todos los elementos en un array 10x10 lleno de unos.
    # -------------------------------------------------------------------------
    def punto_2(self):
        array_unos = np.ones((10, 10))
        suma_total = np.sum(array_unos)
        self.df.loc[len(self.df)] = [
            "2. Suma en array 10x10 de unos",
            suma_total
        ]
    
    # -------------------------------------------------------------------------
    # 3. Producto elemento a elemento de dos arrays de tamaño 5 (aleatorios 1..10).
    # -------------------------------------------------------------------------
    def punto_3(self):
        arr1 = np.random.randint(1, 11, size=5)
        arr2 = np.random.randint(1, 11, size=5)
        producto = arr1 * arr2
        resultado_str = f"arr1={arr1}, arr2={arr2}, producto={producto}"
        self.df.loc[len(self.df)] = [
            "3. Producto elemento a elemento (size=5)",
            resultado_str
        ]
    
    # -------------------------------------------------------------------------
    # 4. Matriz 4x4 (elemento = i+j) y su inversa (o pseudo-inversa si es singular).
    # -------------------------------------------------------------------------
    def punto_4(self):
        matriz_4x4 = np.fromfunction(lambda i, j: i + j, (4, 4), dtype=int)
        det = np.linalg.det(matriz_4x4)
        if abs(det) < 1e-12:
            inversa = np.linalg.pinv(matriz_4x4)
            tipo_inv = "Pseudo-inversa (matriz singular)"
        else:
            inversa = np.linalg.inv(matriz_4x4)
            tipo_inv = "Inversa"
        resultado_str = (
            f"Matriz:\n{matriz_4x4}\n\n"
            f"Determinante: {det}\n"
            f"{tipo_inv}:\n{inversa}"
        )
        self.df.loc[len(self.df)] = [
            "4. Matriz 4x4 (i+j) e inversa",
            resultado_str
        ]
    
    # -------------------------------------------------------------------------
    # 5. Máximo y mínimo en un array de 100 elementos aleatorios, mostrar índices.
    # -------------------------------------------------------------------------
    def punto_5(self):
        arr_100 = np.random.rand(100)
        max_val = np.max(arr_100)
        min_val = np.min(arr_100)
        max_idx = np.argmax(arr_100)
        min_idx = np.argmin(arr_100)
        resultado_str = (
            f"Array:\n{arr_100}\n\n"
            f"Valor máximo = {max_val} en índice {max_idx}\n"
            f"Valor mínimo = {min_val} en índice {min_idx}"
        )
        self.df.loc[len(self.df)] = [
            "5. Máximo y mínimo en array de 100 elementos",
            resultado_str
        ]
    
    # -------------------------------------------------------------------------
    # 6. Sumar un array 3x1 y uno 1x3 (broadcasting) -> array 3x3.
    # -------------------------------------------------------------------------
    def punto_6(self):
        arr_3x1 = np.array([[1], [2], [3]])
        arr_1x3 = np.array([[10, 20, 30]])
        suma_broadcast = arr_3x1 + arr_1x3
        resultado_str = (
            f"Array 3x1:\n{arr_3x1}\n\n"
            f"Array 1x3:\n{arr_1x3}\n\n"
            f"Suma (broadcast):\n{suma_broadcast}"
        )
        self.df.loc[len(self.df)] = [
            "6. Broadcasting 3x1 + 1x3",
            resultado_str
        ]
    
    # -------------------------------------------------------------------------
    # 7. De una matriz 5x5, extraer una submatriz 2x2 que comience en la 2a fila y columna.
    # -------------------------------------------------------------------------
    def punto_7(self):
        matriz_5x5 = np.arange(1, 26).reshape((5, 5))
        submatriz = matriz_5x5[1:3, 1:3]
        resultado_str = (
            f"Matriz 5x5:\n{matriz_5x5}\n\n"
            f"Submatriz 2x2 (desde fila=1, col=1):\n{submatriz}"
        )
        self.df.loc[len(self.df)] = [
            "7. Submatriz 2x2 en 5x5 (2a fila, 2a col)",
            resultado_str
        ]
    
    # -------------------------------------------------------------------------
    # 8. Crear un array de ceros (size=10) y cambiar índices 3..6 a 5.
    # -------------------------------------------------------------------------
    def punto_8(self):
        arr_ceros = np.zeros(10)
        arr_ceros[3:7] = 5
        self.df.loc[len(self.df)] = [
            "8. Array ceros (size=10) c/ índices 3..6 = 5",
            str(arr_ceros)
        ]
    
    # -------------------------------------------------------------------------
    # 9. Invertir el orden de las filas en una matriz 3x3.
    # -------------------------------------------------------------------------
    def punto_9(self):
        matriz_3x3 = np.array([[1, 2, 3],
                               [4, 5, 6],
                               [7, 8, 9]])
        matriz_invertida = matriz_3x3[::-1]
        resultado_str = (
            f"Matriz original:\n{matriz_3x3}\n\n"
            f"Matriz invertida (filas):\n{matriz_invertida}"
        )
        self.df.loc[len(self.df)] = [
            "9. Invertir filas en matriz 3x3",
            resultado_str
        ]
    
    # -------------------------------------------------------------------------
    # 10. Seleccionar valores > 0.5 de un array aleatorio de tamaño 10.
    # -------------------------------------------------------------------------
    def punto_10(self):
        arr_rand10 = np.random.rand(10)
        mayores = arr_rand10[arr_rand10 > 0.5]
        resultado_str = (
            f"Array:\n{arr_rand10}\n\n"
            f"Elementos > 0.5:\n{mayores}"
        )
        self.df.loc[len(self.df)] = [
            "10. Elementos mayores a 0.5 (array size=10)",
            resultado_str
        ]
    
    # -------------------------------------------------------------------------
    # 11. Gráfico de dispersión con dos arrays de tamaño 100.
    # -------------------------------------------------------------------------
    def punto_11(self):
        x = np.random.rand(100)
        y = np.random.rand(100)
        plt.figure()
        plt.scatter(x, y, c='blue', label='Puntos aleatorios')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('11. Gráfico de dispersión (100 puntos aleatorios)')
        plt.legend()
        plt.savefig("punto_11_scatter.png")
        plt.close()
        self.df.loc[len(self.df)] = [
            "11. Gráfico de dispersión (100 puntos)",
            "Archivo: punto_11_scatter.png"
        ]
    
    # -------------------------------------------------------------------------
    # 12. Gráfico de dispersión: x en [-2π, 2π], y = sin(x) + ruido gaussiano; graficar también y=sin(x).
    # -------------------------------------------------------------------------
    def punto_12(self):
        x_vals = np.linspace(-2*np.pi, 2*np.pi, 100)
        y_sin = np.sin(x_vals)
        ruido = np.random.normal(0, 0.2, x_vals.shape)
        y_ruido = y_sin + ruido
        plt.figure()
        plt.scatter(x_vals, y_ruido, color='red', label='sin(x) + ruido')
        plt.plot(x_vals, y_sin, color='green', label='sin(x)')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('12. sin(x) + ruido vs sin(x)')
        plt.legend()
        plt.savefig("punto_12_sin_ruido.png")
        plt.close()
        self.df.loc[len(self.df)] = [
            "12. sin(x)+ruido y sin(x)",
            "Archivo: punto_12_sin_ruido.png"
        ]
    
    # -------------------------------------------------------------------------
    # 13. Usar np.meshgrid para z = cos(x) + sin(y) y mostrar un gráfico de contorno.
    # -------------------------------------------------------------------------
    def punto_13(self):
        x = np.linspace(-2*np.pi, 2*np.pi, 100)
        y = np.linspace(-2*np.pi, 2*np.pi, 100)
        X, Y = np.meshgrid(x, y)
        Z = np.cos(X) + np.sin(Y)
        plt.figure()
        cont = plt.contour(X, Y, Z, levels=20, cmap='viridis')
        plt.colorbar(cont)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('13. Contorno: cos(x) + sin(y)')
        plt.savefig("punto_13_contour.png")
        plt.close()
        self.df.loc[len(self.df)] = [
            "13. Contorno cos(x)+sin(y)",
            "Archivo: punto_13_contour.png"
        ]
    
    # -------------------------------------------------------------------------
    # 14. Gráfico de dispersión con 1000 puntos, coloreados según densidad.
    # -------------------------------------------------------------------------
    def punto_14(self):
        x_1000 = np.random.rand(1000)
        y_1000 = np.random.rand(1000)
        xy = np.vstack([x_1000, y_1000])
        densidad = gaussian_kde(xy)(xy)
        plt.figure()
        plt.scatter(x_1000, y_1000, c=densidad, s=20, cmap='jet')
        plt.colorbar(label='Densidad')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('14. Dispersión 1000 pts (colores por densidad)')
        plt.savefig("punto_14_density_scatter.png")
        plt.close()
        self.df.loc[len(self.df)] = [
            "14. Scatter 1000 puntos, densidad",
            "Archivo: punto_14_density_scatter.png"
        ]
    
    # -------------------------------------------------------------------------
    # 15. A partir de la misma función (cos(x)+sin(y)), generar un gráfico de contorno lleno.
    # -------------------------------------------------------------------------
    def punto_15(self):
        x = np.linspace(-2*np.pi, 2*np.pi, 100)
        y = np.linspace(-2*np.pi, 2*np.pi, 100)
        X, Y = np.meshgrid(x, y)
        Z = np.cos(X) + np.sin(Y)
        plt.figure()
        cont_filled = plt.contourf(X, Y, Z, levels=20, cmap='viridis')
        plt.colorbar(cont_filled)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('15. Contorno lleno: cos(x) + sin(y)')
        plt.savefig("punto_15_filled_contour.png")
        plt.close()
        self.df.loc[len(self.df)] = [
            "15. Contorno lleno cos(x)+sin(y)",
            "Archivo: punto_15_filled_contour.png"
        ]
    
    # -------------------------------------------------------------------------
    # 16. Añadir etiquetas Eje X, Eje Y y título a dispersión del ejercicio 12, con leyendas en LaTeX.
    # -------------------------------------------------------------------------
    def punto_16(self):
        x_vals = np.linspace(-2*np.pi, 2*np.pi, 100)
        y_sin = np.sin(x_vals)
        ruido = np.random.normal(0, 0.2, x_vals.shape)
        y_ruido = y_sin + ruido
        plt.figure()
        plt.scatter(x_vals, y_ruido, color='red', label=r'$y = \sin(x) + \text{ruido}$')
        plt.plot(x_vals, y_sin, color='blue', label=r'$y = \sin(x)$')
        plt.xlabel(r'Eje X')
        plt.ylabel(r'Eje Y')
        plt.title(r'16. Gráfico de Dispersión')
        plt.legend()
        plt.savefig("punto_16_latex_labels.png")
        plt.close()
        self.df.loc[len(self.df)] = [
            "16. Dispersión con etiquetas y leyendas LaTeX",
            "Archivo: punto_16_latex_labels.png"
        ]
    
    # -------------------------------------------------------------------------
    # 17. Crear un histograma con 1000 números aleatorios de distribución normal.
    # -------------------------------------------------------------------------
    def punto_17(self):
        datos_normal = np.random.randn(1000)
        plt.figure()
        plt.hist(datos_normal, bins=30, alpha=0.7, color='blue')
        plt.title('17. Histograma (Normal N(0,1))')
        plt.xlabel('Valores')
        plt.ylabel('Frecuencia')
        plt.savefig("punto_17_hist_normal.png")
        plt.close()
        self.df.loc[len(self.df)] = [
            "17. Histograma N(0,1)",
            "Archivo: punto_17_hist_normal.png"
        ]
    
    # -------------------------------------------------------------------------
    # 18. Dos sets de datos con distribuciones normales distintas, en el mismo histograma.
    # -------------------------------------------------------------------------
    def punto_18(self):
        datos1 = np.random.normal(0, 1, 1000)
        datos2 = np.random.normal(2, 0.5, 1000)
        plt.figure()
        plt.hist(datos1, bins=30, alpha=0.5, label='N(0,1)', color='blue')
        plt.hist(datos2, bins=30, alpha=0.5, label='N(2,0.5)', color='red')
        plt.title('18. Dos distribuciones normales en un histograma')
        plt.xlabel('Valores')
        plt.ylabel('Frecuencia')
        plt.legend()
        plt.savefig("punto_18_dos_normales.png")
        plt.close()
        self.df.loc[len(self.df)] = [
            "18. Dos distribuciones normales",
            "Archivo: punto_18_dos_normales.png"
        ]
    
    # -------------------------------------------------------------------------
    # 19. Experimentar con bins (10, 30, 50) en un histograma y observar los cambios.
    # -------------------------------------------------------------------------
    def punto_19(self):
        datos = np.random.randn(1000)
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        axs[0].hist(datos, bins=10, color='green', alpha=0.7)
        axs[0].set_title('Bins=10')
        axs[1].hist(datos, bins=30, color='green', alpha=0.7)
        axs[1].set_title('Bins=30')
        axs[2].hist(datos, bins=50, color='green', alpha=0.7)
        axs[2].set_title('Bins=50')
        fig.suptitle('19. Histograma con bins=10,30,50')
        plt.savefig("punto_19_distintos_bins.png")
        plt.close()
        self.df.loc[len(self.df)] = [
            "19. Histograma con distintos bins",
            "Archivo: punto_19_distintos_bins.png"
        ]
    
    # -------------------------------------------------------------------------
    # 20. Añadir una línea vertical que indique la media en el histograma.
    # -------------------------------------------------------------------------
    def punto_20(self):
        datos = np.random.randn(1000)
        media = np.mean(datos)
        plt.figure()
        plt.hist(datos, bins=30, alpha=0.7, color='purple')
        plt.axvline(media, color='red', linestyle='dashed', linewidth=2,
                    label=f'Media={media:.2f}')
        plt.title('20. Histograma con línea de la media')
        plt.xlabel('Valores')
        plt.ylabel('Frecuencia')
        plt.legend()
        plt.savefig("punto_20_linea_media.png")
        plt.close()
        self.df.loc[len(self.df)] = [
            "20. Histograma con línea de media",
            f"Archivo: punto_20_linea_media.png (Media={media:.2f})"
        ]
    
    # -------------------------------------------------------------------------
    # 21. Crear histogramas superpuestos con colores y transparencias.
    # -------------------------------------------------------------------------
    def punto_21(self):
        datos1 = np.random.normal(0, 1, 1000)
        datos2 = np.random.normal(2, 0.5, 1000)
        plt.figure()
        plt.hist(datos1, bins=30, color='blue', alpha=0.5, label='Dist 1')
        plt.hist(datos2, bins=30, color='red', alpha=0.5, label='Dist 2')
        plt.title('21. Histogramas superpuestos')
        plt.xlabel('Valores')
        plt.ylabel('Frecuencia')
        plt.legend()
        plt.savefig("punto_21_hist_superpuestos.png")
        plt.close()
        self.df.loc[len(self.df)] = [
            "21. Histogramas superpuestos",
            "Archivo: punto_21_hist_superpuestos.png"
        ]
    
    # -------------------------------------------------------------------------
    # Método para exportar el DataFrame a Excel con imágenes incrustadas
    # -------------------------------------------------------------------------
    def guardar_excel(self, nombre_archivo="actividad2_con_graficos.xlsx"):
        """
        Exporta el DataFrame a un archivo Excel (usando xlsxwriter) e inserta
        las imágenes (PNG) referenciadas en la columna 'Resultado' en columnas
        adyacentes, ajustando filas y columnas para evitar superposición.
        """
        with pd.ExcelWriter(nombre_archivo, engine='xlsxwriter') as writer:
            # Escribe el DataFrame en la hoja "Resultados"
            self.df.to_excel(writer, sheet_name='Resultados', index=False)
            
            workbook  = writer.book
            worksheet = writer.sheets['Resultados']

            # Ajustar el ancho de las columnas (D..K) para que quepan las imágenes
            worksheet.set_column(3, 10, 35)  # Columnas 3..10 => D..K, ancho=35 (ajusta a tu gusto)

            # Ajustar la altura de todas las filas para dar más espacio vertical
            for row_i in range(1, len(self.df) + 2):
                worksheet.set_row(row_i, 80)  # 80 puntos de altura (ajusta según necesites)

            # Recorrer cada fila y, si en "Resultado" hay "Archivo:", insertar la(s) imagen(es).
            for i in range(len(self.df)):
                result_str = str(self.df.loc[i, "Resultado"])
                if "Archivo:" in result_str:
                    splitted = result_str.split("Archivo:")
                    image_files = [part.strip().split()[0] for part in splitted if part.strip() != ""]
                    
                    col_offset = 0  # Comenzar a partir de la columna D (índice 3)
                    for image_file in image_files:
                        try:
                            worksheet.insert_image(
                                i + 1,            # Fila donde insertar la imagen
                                3 + col_offset,   # Columna D + desplazamiento
                                image_file,
                                {
                                    'x_scale': 0.3,
                                    'y_scale': 0.3,
                                    'x_offset': 10,
                                    'y_offset': 10
                                }
                            )
                            col_offset += 1
                        except Exception as e:
                            print(f"Error al insertar la imagen {image_file} en la fila {i+1}: {e}")

        print(f"Archivo Excel '{nombre_archivo}' con gráficos incrustados generado con éxito.")

# ------------------------------------------------------------------------------
# EJECUCIÓN de todos los ejercicios y exportación a Excel
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    ejercicios = Ejercicios()
    ejercicios.punto_1()
    ejercicios.punto_2()
    ejercicios.punto_3()
    ejercicios.punto_4()
    ejercicios.punto_5()
    ejercicios.punto_6()
    ejercicios.punto_7()
    ejercicios.punto_8()
    ejercicios.punto_9()
    ejercicios.punto_10()
    ejercicios.punto_11()
    ejercicios.punto_12()
    ejercicios.punto_13()
    ejercicios.punto_14()
    ejercicios.punto_15()
    ejercicios.punto_16()
    ejercicios.punto_17()
    ejercicios.punto_18()
    ejercicios.punto_19()
    ejercicios.punto_20()
    ejercicios.punto_21()
    ejercicios.guardar_excel("actividad2_con_graficos.xlsx")

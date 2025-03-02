import json
import requests
import sys 

class Actividad_1():
    def __init__(self):
        self.ruta_static="src/pad/static"
        sys.stdout.reconfigure(encoding='utf-8')  

    def leer_api(self,url):
        response=requests.get(url)
        return response.json()
        
    def escribir_json(self,nombre_archivo, datos_json):
       ruta_json= f"{self.ruta_static}/json/{nombre_archivo}"
       try:
        
        with open(ruta_json,"w", encoding="utf-8") as f:
            json.dump(datos_json,f,ensure_ascii=False,indent=4)
            print(f"El archivo {nombre_archivo} se ha creado correctamente en JSON")
            return True
        
       except Exception as e:
           print(f"Error al escribir el archivo {nombre_archivo} en JSON")
           print(e)
           return False
          


    def escribir_txt(self, nombre_archivo, datos=None):
        if nombre_archivo=="":
            nombre_archivo="datos.txt"
        if datos is None:
            datos="No se ingresaron datos"

        ruta_txt= f"{self.ruta_static}/txt/{nombre_archivo}"

        with open(ruta_txt,"w", encoding="utf-8") as f:
            
            f.write(datos)
        return True
     
        
#creamos una instancia de la clase Actividad_1
ingestiones=Actividad_1()
# Para obtener la división administrativa de un país en particular, simplemente realice una solicitud GET a
datos_json = ingestiones.leer_api("https://rawcdn.githack.com/kamikazechaser/administrative-divisions-db/master/api/KE.json")
print("datos_json:",datos_json)
if ingestiones.escribir_json("datos.json",datos_json):
    print("El archivo se ha creado correctamente en JSON")



print("esta es la ruta statica :",ingestiones.ruta_static)
print("datos json:",datos_json) 
    

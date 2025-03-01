import json
import requests


class Ingestiones():
    def __init__(self):
        self.ruta_static="src/pad/static/"

    def leer_api(self,url):
        response=requests.get(url)
        return response.json()
        
    def escribir_json(self):
       pass


#creamos una instancia de la clase Ingestiones
ingestiones=Ingestiones()
# Para obtener la división administrativa de un país en particular, simplemente realice una solicitud GET a
datos_json = ingestiones.leer_api("https://rawcdn.githack.com/kamikazechaser/administrative-divisions-db/master/api/KE.json")
print("esta es la ruta statica :",ingestiones.ruta_static)
print("datos json:",datos_json) 
    
# <h1 align="center">¿TODOS LOS VINOS SABEN IGUAL?</h1>

![picture alt](https://cdn.businessinsider.es/sites/navi.axelspringer.es/public/media/image/2021/08/copas-2446819.jpg?tf=3840x)

En este Repositorio, se analizó la distribución de diferentes atributos químicos que contienen los vinos en cultivos de Italia. Se aplicaron anális de diversos tipos y se realizó una comparación entre dos tipos de Clustering: Kmeans y Hierarchical. Se determinó que el mejor por Silhouette Score fue Kmeans. Finalmente, se Dockerizó el modelo en una Api de Flask en Docker para futuras contribuciones =). 


Para correr la API (ya se encuentra subida en Docker Hub):
````docker run -p3000:3000 alexcanchari/mlapp````

Si usas la terminal puedes usar este código o apoyarte de Postman:
`curl -X POST -H "Content-Type:application/json" -d @data.json http://localhost:3000/predict`

Igual por si acaso, está también la lista de requirements:
`pip install -r requirements.txt`


### Dataset

* **Alcohol:** El contenido de alcohol contribuye a las propiedades organolépticas del vino, como el sabor y la sensación en boca.
* **Malic acid:** El ácido málico afecta la acidez y el sabor del vino. Altos niveles pueden contribuir a sabores ácidos y frescos.
* **Ash:** La ceniza proporciona información sobre los minerales presentes en el vino, lo cual puede influir en su sabor y textura.
* **Alcalinity of ash:** Indica el nivel de alcalinidad, lo cual puede estar relacionado con la acidez y la capacidad del vino para neutralizar ácidos.
* **Magnesium:** El magnesio puede influir en la estructura y la textura del vino.
* **Total phenols:** Los fenoles tienen propiedades antioxidantes y pueden contribuir a la coloración, el sabor y la longevidad del vino.
* **Flavanoids:** Los flavonoides también tienen propiedades antioxidantes y pueden contribuir a la estructura y el sabor del vino.
* **Nonflavanoid phenols:** Otro grupo de fenoles con propiedades específicas que pueden afectar el perfil de sabor y la textura del vino.
* **Proanthocyanins:** Las proantocianidinas son compuestos fenólicos que contribuyen a la astringencia y a la estructura del vino.
* **Color intensity:** La intensidad del color puede estar relacionada con la concentración de compuestos fenólicos y flavonoides, afectando la apariencia visual del vino.
* **Hue:** El tono del color puede proporcionar información sobre la edad y el estilo del vino.
* **OD280/OD315 of diluted wines:** Puede proporcionar información sobre la concentración de compuestos específicos y la calidad del vino.
* **Proline:** La proline puede estar relacionada con la madurez de la uva y afectar la estructura y el sabor del vino.



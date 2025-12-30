# LSA-X
Una version extendida del dataset LSA-T

Los canales usados fueron:
- CN Sordos
- Videolibros.org(su canal y sus libros)
- AsociacionCivil

Otros canales que podria usar serian:
- [Locufre](https://www.youtube.com/channel/UCPJr7e9V_07DAID60F0pXVw) si logro separa entre señantes
- [Confederacion argentina de sordos](https://www.youtube.com/watch?v=2GdI4FUxRgo) parece que ningun video tiene audio o subtitulos, pero es mucho contenido(unlabeled data) y lo puedo usar

Bugs:
- Hay veces que alguien se va de frame y el visualize_landmarks_processing se frena
- La escala de las manos anda para el culo, creo que es porque la posicion z de los pose landmarks anda mal

390 horas de video, 20 veces mas que LSA-T

El pipeline de procesado de datos es:
1. Genero la lista de videos a descargar haciendo scraping del sitio de videolibros
2. Descargo todos los videos publicos de los canales de youtube + la lista de URLs scrapeada
3. Separo los videos en audio y video
4. Separo a los videos en unlabeled y labeled
   5. A los videos de CNSordos y de Locufre, como todos los que son labeled tienen subtitulos, simplemente me fijo si su archivo de subtitulos existe o no
   6. Para los otros canales, como no tienen subtitulos y hay que generarlos, uso VAD para fijarme si tienen speech o no
7. Subo los archivos de audio no subtitulados y labeled(todos menos Locufre y CNSordos) a una VM para subtitularlos usando el notebook de create_subtitles
   8. En Locufre hago la excepcion de usar los subtitulos automaticos de youtube en vez de generarlos yo. Eso es porque la mayoria del tiempo tienen audio de muy alta calidad(estan en un estudio) y por la forma en que son los datos, no se si vale la pena invertir el tiempo y plata en generarle subtitulos de alta calidad
8. Subo todos los videos(labeled o unlabeled) a una VM para generar los bounding_boxes usando YOLO
9. Genero los landmarks de cuerpo, mano y cara para todos(labeled o unlabeled)
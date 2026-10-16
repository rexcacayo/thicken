# Thicken 1.1.0

Complemento para Blender 4.2 a 5.x. **Engorda solo las zonas demasiado finas** para imprimir (pestañas, bordes, dedos, capas) hasta el grosor mínimo de tu boquilla. El resto del modelo no se toca.


## Qué cambia en 1.1.0

- Nuevo modo **Solo grosor**: engorda a lo largo de un eje **sin ensanchar**. La silueta vista desde ese
  eje queda exacta, así una incrustación, un logo, una aleta o una placa siguen entrando en su hueco.
  - **Eje**: Auto (la medida más pequeña de la pieza) o X / Y / Z.
  - **Crece**: *solo hacia arriba* (la cara de abajo no se mueve: pegada a la cama o al fondo del hueco)
    o *por las dos caras*.
  - El grosor final es el **Grosor mínimo** (o dos pasadas de tu boquilla si lo dejas en 0).
- El modo de siempre se llama ahora **Zonas finas**.
## Uso (pestaña Thicken)

1. Selecciona el modelo, elige tu boquilla (o escribe el grosor mínimo) y pulsa **Buscar zonas finas**: rojo = no sale, ámbar = frágil.
2. **Engordar zonas finas**: crea `<modelo>_grueso`; cada pared fina crece por sus dos caras hasta el mínimo, con una transición suave. Colores conservados; el original queda oculto.
3. Pásalo después por **Mesh Doctor** por si alguna zona engordada roza con otra.

Orden recomendado: Mesh Doctor → Print Scale → **Thicken** → Mesh Doctor → resto.

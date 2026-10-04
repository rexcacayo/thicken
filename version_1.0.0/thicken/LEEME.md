# Thicken 1.0.0

Complemento para Blender 4.2 a 5.x. **Engorda solo las zonas demasiado finas** para imprimir (pestañas, bordes, dedos, capas) hasta el grosor mínimo de tu boquilla. El resto del modelo no se toca.

## Uso (pestaña Thicken)

1. Selecciona el modelo, elige tu boquilla (o escribe el grosor mínimo) y pulsa **Buscar zonas finas**: rojo = no sale, ámbar = frágil.
2. **Engordar zonas finas**: crea `<modelo>_grueso`; cada pared fina crece por sus dos caras hasta el mínimo, con una transición suave. Colores conservados; el original queda oculto.
3. Pásalo después por **Mesh Doctor** por si alguna zona engordada roza con otra.

Orden recomendado: Mesh Doctor → Print Scale → **Thicken** → Mesh Doctor → resto.

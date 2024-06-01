# Feature-Learning
Nuevas clases:

    BMBClassifier, ESClassifier, ILSClassifier y ILS_ESClassifier

    ILS implementa la busqueda local y el operador de vecino común.

    BMB hereda de ILS para no repetir código.

    ES va aparte.

    ILS-ES hereda de ILS para aplicar el mismo esquema de iteración que ILS pero llamando a una nueva funcion "ESIndividual" que aplica ES sobre una solucion concreta en lugar de generar una inicial aleatoria.

NOTA IMPORTANTE: 
    En lugar de implementar un funcion con nombre ILS-ES, como ILS-ES hereda de de ILS, simplemente hago que la funcion ILS reciba una funcion como argumento, y cuando tiene que optimizar una solucion con BL, le paso busquedaLocalSolucion como argumento y cuando tengo que implementar ILS-ES le paso ESIndividual como argumento. Vease memoria y pseudocódigos.
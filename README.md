# Feature-Learning
En mi código no implemento todos los métodos/funciones solicitados sino que implemento una estructura de herencia de clases para no repetir código.

Para los algoritmos genéticos, en lugar de tener los métodos/funciones:
    - AGG
    - AGE

He definido la clase GeneticClassifier para encapsular el esquema de evolución común tanto del AGG como del AGE en un método llamado:
    - AG. 
    
Además, en GeneticClassifier implemento los métodos siguientes que también son comunes a ambos (AGG y AGE):
    - evaluarPoblacion
    - mejorIndividuo
    - generarPadres
    - seleccionTorneo
    - cruce
    - cruceBLX
    - cruceAritmetico

Luego, defino las clases AGGClassifier y AGEClassifier que heredan de GeneticClassifier e implementan solamente los métodos que difieren en ambos esquemas:
    - mutacion
    - reemplazo

Por lo tanto, para implementar un clasificador AGG o AGE, en lugar de llamar a AGG/AGE, llamo a AG y esta se encarga de llamar a las funciones de mutacion/reemplazo correspondientes según la clase del objeto clasificador creado.

Para los algoritmos meméticos lo hago de manera análoga, en lugar de tener los métodos/funciones:
    - AM-All
    - AM-Rand

He definido la clase AMProbClassifier y el método AM_Prob para encapsular tanto el calasificador AM-ALL como el AM-RAND en un clasificador padre donde varía la cantidad de población escogida para la BL según una probabilidad como parámetro. Si la probabilidad de AMProbClassifier es 1.0, entonces el clasificador resultante será un AM-ALL y si la probabilidad de parámetro es 0.1, entonces el clasificador resultante será AM-RAND.
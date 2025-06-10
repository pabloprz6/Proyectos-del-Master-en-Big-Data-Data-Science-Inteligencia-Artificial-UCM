#!/usr/bin/env python
# coding: utf-8

# In[2]:


from mrjob.job import MRJob

class MR_movies(MRJob):

    def mapper(self, _, line):
        
        campos = line.strip().split("|")
        idioma = campos[2]
        pais = campos[3]
        presupuesto = campos[4]

        if idioma and pais and presupuesto.isdigit():
            
            yield idioma, (pais, int(presupuesto))

    def reducer(self, idioma, pais_presupuesto):
        
        paises = []
        presupuesto_total = 0
        
        # Añadimos los paises sin duplicados y calculamos el presupuesto total para cada idioma
        for pais, presupuesto in pais_presupuesto:
            if pais not in paises:
                paises.append(pais)
            presupuesto_total += presupuesto
        
        yield idioma, [paises, presupuesto_total]
        
if __name__ == '__main__':
    MR_movies.run()


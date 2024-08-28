# Opis projekta

To je moje diplomsko delo z naslovom "**Napredne s poizvedovanjem obogatene tehnike generiranja (RAG)**" (angl. Advanced retrieval augmented generation techniques). V njem raziščem kako uspešne so različne metode kot pravni asistent. Raziskane so naslednje metode:

1. velik jezikovni model (VJM)
2. naivni RAG in VJM
3. napredni RAG in VJM
4. modularni RAG in VJM

Za podrobnosti metod si preberite [četrto poglavje diplome](diploma.pdf) ali pa si oglejte [izvorno kodo](src/chains.py).

# Struktura repozitorija

- Vsa izvorna koda je dostopna v direktoriju [src](src).
- Pdf diplome si je moč ogledati v datoteki [diploma.pdf](diploma.pdf).
- V [direktoriju s testnimi podatki](test_data) si lahko ogledate vsa vprašanja in odgovore na ta vprašanja na katerih so metode bile testirane. Zraven si lahko v Excel razpredelnici ogledate še ocene metod za različne scenarije, ki jih je določil strokovnjak.
- V direktoriju [generated_outputs](generated_outputs) si lahko ogledate vse odgovore, ki so jih zgenerirale metode.

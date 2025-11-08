# Centroid

Visualisation audio réactive en 3D inspirée des constellations de Lucio Arese.

## Prérequis

- Python 3.9 ou supérieur
- [PortAudio](http://www.portaudio.com/) installé sur le système (requis par `sounddevice`)

Installez ensuite les dépendances Python :

```bash
pip install -r requirements.txt
```

## Utilisation

Lancez la visualisation avec :

```bash
python main.py
```

Par défaut, l’application capture le signal du micro par défaut et projette en
3D un nuage de carrés connectés qui matérialise l’évolution timbrale du son.

### Contrôles

La caméra 3D est une caméra orbitale : utilisez la souris pour tourner autour
de la scène et la molette pour zoomer/dézoomer.

## Description artistique

À chaque trame audio, le script extrait :

- **Spectral Centroid**
- **Spectral Spread**
- **Spectral Flatness** (utilisé pour calculer la *Tonality* = 1 − flatness)
- **Spectral Flux** (flux spectral normalisé)
- **RMS** lissé

Ces paramètres contrôlent les axes et l’apparence des éléments visuels :

- `X = Tonality`
- `Y = Spectral Centroid` normalisé
- `Z = Spectral Spread` normalisé
- La couleur et la taille suivent le *Spectral Flux* pour révéler l’énergie des
  attaques.
- L’alpha décroît avec l’âge pour créer un effet de traînée poétique.

Les points sont reliés temporellement afin de dessiner une trajectoire
tridimensionnelle épaissie par les pics de flux, produisant des rubans et des
filaments vibrants.

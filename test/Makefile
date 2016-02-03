#
#	Makefile do projeto foveatedFeatures
#

#
# A variável CC indica que compilador que será utilizado
#
CC= g++

#
# A variável CFLAGS indica que opção de compilação queremos
#
CFLAGS= -Wall -pedantic -ansi -g -c

#
# A variável LFLAGS indica que opção de compilação queremos
#
LFLAGS= -Wall -g

#
# A variável DIR indica o diretório dos arquivos de cabeçalho do projeto
#
DIR= ~/foveatedFeatures

#
# A variável INCS indica o caminho dos arquivos de cabeçalho
#
INCS= -I../ -I/usr/local/include/opencv2 -I/usr/local/include

#
# A variável LIBS indica o caminho das bibliotecas e as bibliotecas usadas na ligação
#
LIBS=

#
# A variável OBJS indica os arquivos objetos comuns a todos os executáveis
#
OBJS= 

#
# A variável ALL indica os arquivos executáveis
#
all: match extract tracking

#
# Regra de compilação e ligação do executáveis
# 
match:	match.cpp $(DIR)/foveatedHessianDetector.h
	$(CC) $(LFLAGS) match.cpp `pkg-config --cflags --libs opencv` -o match $(LIBS) -O3

#
# Regra de compilação e ligação do executáveis
# 
extract: extract.cpp $(DIR)/foveatedHessianDetector.h
	$(CC) $(LFLAGS) extract.cpp `pkg-config --cflags --libs opencv` -o extract $(LIBS) -O3

#
# Regra de compilação e ligação do executáveis
# 
tracking: tracking.cpp $(DIR)/foveatedHessianDetector.h $(DIR)/foveatedTracking.h
	$(CC) $(LFLAGS) tracking.cpp `pkg-config --cflags --libs opencv` -o tracking $(LIBS) -O3

#
# Regra que elimina arquivos intermediários
# 
clean:
	rm -f *.o *.cpp~

#
# Regra que elimina arquivos intermediários e executável
# 
realclean:
	rm -f *.o *.cpp~ match extract tracking


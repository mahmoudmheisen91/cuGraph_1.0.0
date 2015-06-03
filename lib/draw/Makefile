CXX = g++
MOCQT4 = moc-qt4
FILES := draw.cpp

all: draw.o polygon

clean:
	rm -f polygon.png polygon draw.o

INCL = -I/usr/include/qt4 -I/usr/include/qt4/QtCore -I/usr/include/qt4/QtGui
FLAGS =  -lQtCore -lQtGui

draw.o: $(FILES) draw.h
	$(MOCQT4) $(FILES) | $(CXX) $(INCL) -c -x c++ - -include $(FILES) -o draw.o

polygon: polygon.cpp draw.o draw.h
	$(CXX) $@.cpp draw.o $(FLAGS) -o $@
	
	

CXX = g++
MOCQT4 = moc-qt4

EXAMPLES = main
all: Editor.o $(EXAMPLES)

clean:
	rm -f *~ Editor.o main.o testImage.png $(EXAMPLES)

fresh: clean all

INCL = -I/usr/include/qt4 -I/usr/include/qt4/QtCore -I/usr/include/qt4/QtGui
OFLAGS = $(INCL) -Wall -Wno-unreachable-code -Wno-return-type
FLAGS = -g -Wall -lQtCore -lQtGui

Editor.o: Editor.cpp Editor.h
	$(PATCH)
	$(MOCQT4) Editor.h | $(CXX) $(OFLAGS) -c -x c++ - -include Editor.cpp -o Editor.o

%: Editor.o %.cpp Editor.h
	$(CXX) $@.cpp  $(INCL) Editor.o $(FLAGS) -o $@

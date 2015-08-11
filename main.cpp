/*
 *   CUDA Graph Generator.
 *
 *   Copyright (C) 2015 by
 *   Mahmoud Mheisen        mahmoudmheisen91@gmail.com
 *
 *   This program is free software; you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation; either version 2 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 */
 
// Standerd C++ Libraries:
#include <iostream>
#include <string>

// Standerd C Libraries:
#include <cstdio>
#include <omp.h>

// Project Headers:
#include <main/cuGraph.h>

// Namespaces:
using namespace std;					// standerd
using namespace cuGraph;				// project
using namespace CommandLineProcessing;	// parse_options

// Functions Prototypes:
void parse_options(int, char **);		// parse comand line options

// Global Variables:
long int vertix;						// vertix size
long int edge;							// edge size
float prob;								// probability value
string type;							// generator type
string file;							// destination file

// main entry:
int main(int argc, char** argv) 
{
	// Parse command line arguments:
	parse_options(argc, argv);
	
	// Setup the Graph:
    Graph generatedGraph;
    generatedGraph.setType(DIRECTED);
    generatedGraph.setNumberOfVertices(vertix);

	// Start Timer:
	double elapsedTime = omp_get_wtime();
	
	// Choose Generator:
	if (type.compare("ER") == 0) {
		generatedGraph.fillByBaselineER(edge, prob);	// Baseline ER
	} else if (type.compare("ZER") == 0) {
		generatedGraph.fillByZER(edge, prob);			// ZER
	} else if (type.compare("preZER") == 0) {
		generatedGraph.fillByPreZER(edge, prob, 8);		// preZER
	} else if (type.compare("PER") == 0) {
		generatedGraph.fillByPER(edge, prob);			// PER
	} else if (type.compare("PZER") == 0) {
		generatedGraph.fillByPZER(edge, prob, 3);		// PZER
	} else if (type.compare("PPreZER") == 0) {
		generatedGraph.fillByPPreZER(edge, prob, 3, 8); // PPreZER
	}
	
	// Stop Timer:
	elapsedTime = (omp_get_wtime() - elapsedTime) * 1000;
	
	// Print elapsed Time:
    cout << "Elabsed Time to Genrate Graph = " << elapsedTime;
    cout << " ms, Edge count = " << generatedGraph.countEdges() << endl << endl;

	// Save to file:
	ogstream graphStream;
    graphStream.open(file);
    graphStream << generatedGraph;
    graphStream.close();

	// End:
    return 0;
}

/* Parse command line arguments */
void parse_options(int argc, 		/* in */ 
				   char *argv[])	/* in */ 
{
	// Parser object:
	ArgvParser cmd;

  	// Init:
  	cmd.setIntroductoryDescription("cuGraph Command Line Options:");

	// Help option:
  	cmd.setHelpOption("h", "help", "Print this help page.");

	// Other options:
  	cmd.defineOption("vertix"	, "Graph size in term of vertix, default: 10000"		, ArgvParser::OptionRequiresValue);
  	cmd.defineOption("edge"		, "Maximum number of egdes in graph, default: vertix^2"	, ArgvParser::OptionRequiresValue);
  	cmd.defineOption("prob"		, "Probability to choose an edge, default: 0.5"			, ArgvParser::OptionRequiresValue);
  	cmd.defineOption("type"		, "Type of the generator, default: PZER"				, ArgvParser::OptionRequiresValue);
  	cmd.defineOption("file"		, "Destination file, default: output/MTX/graph1.mtx"	, ArgvParser::OptionRequiresValue);
  	
  	// Options alternatives:
  	cmd.defineOptionAlternative("vertix","v");
  	cmd.defineOptionAlternative("edge"	,"e");
  	cmd.defineOptionAlternative("prob"	,"p");
  	cmd.defineOptionAlternative("type"	,"t");
  	cmd.defineOptionAlternative("file"	,"f");

  	// Parse command line:
  	int option = cmd.parse(argc, argv);
  	if (option != ArgvParser::NoParserError) {
    	cout << cmd.parseErrorDescription(option);
    	exit(1);
  	}

  	/* Query the parsing options */
  	
  	// Vertix:
  	if (cmd.foundOption("vertix")) {
    	vertix = std::stoi(cmd.optionValue("vertix"), NULL);
  	} else { // Default value:
  		vertix = 10000;
  	}
  	
  	// Edge:
  	if (cmd.foundOption("edge")) {
    	edge = std::stoi(cmd.optionValue("edge"), NULL);
  	} else { // Default value:
  		edge = vertix * vertix;
  	}
  	
  	// Prob:
  	if (cmd.foundOption("prob")) {
    	prob = std::stoi(cmd.optionValue("prob"), NULL);
  	} else { // Default value:
  		prob = 0.5;
  	}
  	
  	// Type:
  	if (cmd.foundOption("type")) {
    	type = cmd.optionValue("type");
  	} else { // Default value:
  		type = "PZER";
  	}
  	
  	// File:
  	if (cmd.foundOption("file")) {
    	file = cmd.optionValue("file");
  	} else { // Default value:
  		file = "graph1.mtx";
  	}

	// Display Configuration Settings:
	cout << endl;
	cout << "Number of Vertix is set to " << vertix << endl;
	cout << "Number of Edges is set to " << edge << endl;
	cout << "Prob is set to " << prob << endl;
	cout << "Type is set to " << type << endl;
	cout << "File is set to " << file << endl;
	cout << endl;
}



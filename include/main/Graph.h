/**
 * \class Graph
 * \file Graph.cpp
 * \ingroup cuGraph
 *
 * \brief Graph g1;
 * \brief Graph g1(numberOfVertexes);
 *
 * Graph Class: this class represent graph data structure as adjacance matrix
 * rather than adjacance list to make use of the massive parallelism of CUDA platform.
 * The graph can be randomly filled by varous alogorithms, the generation can be 
 * done on CPU or GPU, depending on filling method.
 *
 * $Author: Mahmoud Mheisen $
 *
 * $Revision: 1.0.0 $
 *
 * $Date: 2015/06/14 23:05:20 $
 *
 * $Contact: mahmoudmheisen91@gmail.com $
 *
 * Created on: Fri May 1 18:39:37 2015
 */

#ifndef GRAPH_H_
#define GRAPH_H_
// acycle graph

#include <iostream>
#include <string>
#include "dataTypes.h"

using namespace std;

namespace cuGraph {

    class Graph {
    	/// This class is friend with GraphDraw class to let it make use of the protected member (content)
        friend class GraphDraw;
        
        /// This class is friend with Path class to let it make use of the protected member (content)
        friend class Path;
        
        /// This class is friend with ogstream class to let it make use of the protected member (content)
        friend class ogstream;
        
        /// This class is friend with igstream class to let it make use of the protected member (content)
        friend class igstream;

        public:
        	/** \brief Create an Empty Graph
			  * \see Graph(int numberOfVertices);
			  */
            Graph();
            
            /** \brief Create a Graph with pre defined number of vertices
			  * \param number of vertices
			  * \see Graph();
			  */
            Graph(int numberOfVertices);
            
            /// Virtual method to delete graph content after termination
            virtual ~Graph();

			/** \brief Set the type of the graph
			  * \param dir: (directed/undirected)
			  * \param lp: (has self loops / no self loops)
			  * \return void
			  * \see setNumberOfVertices(int verts);
			  */
            void setType(int dir, int lp);
            
            /** \brief Set number of vertices
			  * \param verts: number of vertices
			  * \return void
			  * This method will clear the graph from all edges and reset the number of vertices
			  * \see clear();
			  */
            void setNumberOfVertices(int verts);

			/** \brief clear the graph from all edges, but keep the vertices
			  * \return void
			  * \see setNumberOfVertices(int verts);
			  */
            void clear(void);
            
            /** \brief add edge between two nodes
			  * \param v1: first node
			  * \param v2: second node
			  * \return void
			  * \see isFull(void);
			  * \see isEmpty(void);
			  */
            void addEdge(int v1, int v2);
            
            /** \brief remove edge between two nodes
			  * \param v1: first node
			  * \param v2: second node
			  * \return void
			  * \see isFull(void);
			  * \see isEmpty(void);
			  */
            void removeEdge(int v1, int v2);

			/** \brief check if the graph is full
			  * \return boolean to indicate if the graph is full or not
			  * Can not add any edge if the graph is full
			  * \see isEmpty(void);
			  * \see isConnected(int v1, int v2);
			  * \see isDirectlyConnected(int v1, int v2);
			  * \see addEdge(int v1, int v2);
			  */
            bool isFull(void);
            
            /** \brief check if the graph is empty
			  * \return boolean to indicate if the graph is empty or not
			  * Can not remove any edge if the graph is empty
			  * \see isFull(void);
			  * \see isConnected(int v1, int v2);
			  * \see isDirectlyConnected(int v1, int v2);
			  * \see removeEdge(int v1, int v2);
			  */
            bool isEmpty(void);
            
            /** \brief check if thier is a path between two nodes using Depth first search
			  * \param v1: first node
			  * \param v2: second node
			  * \return boolean to indicate if their is a path or not
			  * \see isFull(void);
			  * \see isEmpty(void);
			  * \see isDirectlyConnected(int v1, int v2);
			  */
            bool isConnected(int v1, int v2);
            
            /** \brief check if two nodes is directly connected
			  * \param v1: first node
			  * \param v2: second node
			  * \return boolean to indicate if their is an edge or not
			  * \see isFull(void);
			  * \see isEmpty(void);
			  * \see isConnected(int v1, int v2);
			  */
            bool isDirectlyConnected(int v1, int v2);

			/** \brief Serial generator: basic ER
			  * \param E: maximum number of edges
			  * \param p: probability of adding edge to the graph
			  * \return void
              * \see fillByZER(int E, double p);
	          * \see fillByPreLogZER(int E, double p);
	          * \see fillByPreZER(int E, double p, int m);
	          * \see fillByPER(int E, double p);
	          * \see fillByPZER(int E, double p);
	          * \see fillByPPreZER(int E, double p, int m);
			  */
            void fillByBaselineER(int E, double p);
            
            /** \brief Serial generator: ZER
			  * \param E: maximum number of edges
			  * \param p: probability of adding edge to the graph
			  * \return void
			  * \see fillByBaselineER(int E, double p);
	          * \see fillByPreLogZER(int E, double p);
	          * \see fillByPreZER(int E, double p, int m);
	          * \see fillByPER(int E, double p);
	          * \see fillByPZER(int E, double p);
	          * \see fillByPPreZER(int E, double p, int m);
			  */
            void fillByZER(int E, double p);
            
            /** \brief Serial generator: PreLogZER
			  * \param E: maximum number of edges
			  * \param p: probability of adding edge to the graph
			  * \return void
			  * \see fillByBaselineER(int E, double p);
              * \see fillByZER(int E, double p);
	          * \see fillByPreZER(int E, double p, int m);
	          * \see fillByPER(int E, double p);
	          * \see fillByPZER(int E, double p);
	          * \see fillByPPreZER(int E, double p, int m);
			  */
            void fillByPreLogZER(int E, double p);
            
            /** \brief Serial generator: PreZER
			  * \param E: maximum number of edges
			  * \param p: probability of adding edge to the graph
			  * \param m: number of iteration of skipping value, best is 8
			  * \return void
			  * \see fillByBaselineER(int E, double p);
              * \see fillByZER(int E, double p);
	          * \see fillByPreLogZER(int E, double p);
	          * \see fillByPER(int E, double p);
	          * \see fillByPZER(int E, double p);
	          * \see fillByPPreZER(int E, double p, int m);
			  */
            void fillByPreZER(int E, double p, int m);
            
            /** \brief Parallel CUDA generator: PER
			  * \param E: maximum number of edges
			  * \param p: probability of adding edge to the graph
			  * \return void
			  * \see fillByBaselineER(int E, double p);
              * \see fillByZER(int E, double p);
	          * \see fillByPreLogZER(int E, double p);
	          * \see fillByPreZER(int E, double p, int m);
	          * \see fillByPZER(int E, double p);
	          * \see fillByPPreZER(int E, double p, int m);
			  */
            void fillByPER(int E, double p);
            
            /** \brief Parallel CUDA generator: PZER
			  * \param E: maximum number of edges
			  * \param p: probability of adding edge to the graph
			  * \return void
			  * \see fillByBaselineER(int E, double p);
              * \see fillByZER(int E, double p);
	          * \see fillByPreLogZER(int E, double p);
	          * \see fillByPreZER(int E, double p, int m);
	          * \see fillByPER(int E, double p);
	          * \see fillByPPreZER(int E, double p, int m);
			  */
            void fillByPZER(int E, double p);
            
            /** \brief Parallel CUDA generator: PPreZER
			  * \param E: maximum number of edges
			  * \param p: probability of adding edge to the graph
			  * \param m: number of iteration of skipping value, best is 8
			  * \return void
			  * \see fillByBaselineER(int E, double p);
              * \see fillByZER(int E, double p);
	          * \see fillByPreLogZER(int E, double p);
	          * \see fillByPreZER(int E, double p, int m);
	          * \see fillByPER(int E, double p);
	          * \see fillByPZER(int E, double p);
			  */
            void fillByPPreZER(int E, double p, int m);

			/** \brief return number of vertices
			  * \return number of vertices
			  */
            long getNumberOfVertices(void);
            
            /** \brief return number of edges
              * \return number of edges
			  */
            long getNumberOfEdges(void);
            
            /** \brief return type of the graph (directed/undirected)
              * \return type of the graph (directed/undirected)
			  */
            int getDirection(void);
            
            /** \brief return type of the graph (has self loops / no self loops)
              * \return type of the graph (has self loops / no self loops)
			  */
            int getLoop(void);

        protected:
            bool* content; ///< adjacancy matrix of size = v^2

        private:
            bool isInit; 
            long long int  size; 
            long long int  numberOfVertices; 
            long long int  numberOfEdges; 
            int direction; 
            int loop;

            void checkDir(int dir);
            void checkLoop(int lp);
            void checkVertixName(int v1, int v2);
            void checkEdgesBound(int edge);
            void checkVertixesBound(int verts);
            
        }; // end of Graph class
}
#endif // GRAPH_H_














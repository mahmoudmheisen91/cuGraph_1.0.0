#ifndef PATH_H_
#define PATH_H_

class Path
{
	public:
		Path(GraphArray G, int s);
		virtual ~Path();
		bool hasPathTo(int v);
	protected:
	private:
		int s;
		int size;
		bool visited[];
		int edgeTo[];
		void dfs(GraphArray G, int v);
};

#endif // PATH_H_

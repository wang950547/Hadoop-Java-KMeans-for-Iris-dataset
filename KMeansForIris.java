import java.io.IOException;
import java.util.*;
import java.lang.NumberFormatException;
import java.lang.Math;
import java.io.*;
        
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.*;
import org.apache.hadoop.conf.*;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.*;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
        
public class KMeansForIris {
	
//initialization

 public static class InitMap extends Mapper<LongWritable, Text, IntWritable, StringArrayWritable> {

	private int clusterNum;
	private final Random random = new Random();
	private String directory = "KMeans";
	
	protected void setup(Context context) throws IOException, InterruptedException {
		Configuration conf = context.getConfiguration();
        clusterNum =  Integer.parseInt(conf.get("ClusteringNum"));
    }
    //將各點隨機分配到各群    
    public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {

		String line = value.toString();
		
		String[] d = line.split(",");
		if(d!= null && d.length == 5)
			context.write(new IntWritable(1 + random.nextInt(clusterNum)) , new StringArrayWritable(d));
		
    }
 } 
//計算各群群心
 public static class InitReduce extends Reducer<IntWritable, StringArrayWritable, Text, Text> {
	 private int iterate = 0;
	 private int clusterNum;
	 private double[][] centroid ;
	 private String directory = "KMeans";
	 private String[] clusterName ;
	 private String lastPoint = "";
	 //讀出分群數、輸出資料夾
	 protected void setup(Context context) throws IOException, InterruptedException {
        Configuration conf = context.getConfiguration();

		clusterNum =  Integer.parseInt(conf.get("ClusteringNum"));
		centroid = new double[clusterNum][4];
		directory = conf.get("Directory");
		clusterName = new String[clusterNum];
		for(int i = 0; i < clusterName.length; i++)
		 {
			 clusterName[i] = "";
		 }
    }
	//各群計算各自群心
    public void reduce(IntWritable key, Iterable<StringArrayWritable> values, Context context) 
      throws IOException, InterruptedException {
		 double[] center = new double[4];
		 int valSize = 0;
		 for(int i = 0; i < center.length; i++)
		 {
			 center[i] = 0;
		 }
		//把分群結果寫入檔案
		 int k = key.get() - 1;
			for (StringArrayWritable val : values) {
				String[] point = val.toStrings();
				//for centroid
				if(!clusterName[k].contains(point[4]))
				{
					clusterName[k] = clusterName[k] + point[4] + "&";
				}
				
				valSize ++;
				String data = "";
				for(int i = 0; i < point.length - 1; i++)
				{
					data = data + point[i] + ",";
					try{
						center[i] = center[i] + Double.parseDouble(point[i]);
					}catch(Exception e){
						System.out.println(point[i]);
						System.out.println("error data");
					}
				}
				lastPoint = data;
				context.write(new Text(key.toString() + ",") , new Text(data + point[4]));
			}
		//更新群心
		for(int i = 0; i < center.length; i++)
		{
			center[i] = center[i] / valSize;
			centroid[k][i] = center[i];
			System.out.println(center[i]);
		}

    }
	//將更新的群心輸出至檔案
	protected void cleanup(Context context) throws IOException,InterruptedException
	{
		Configuration conf = context.getConfiguration();
		conf.set("lastPoint" , lastPoint );
		updateCentroid(centroid , iterate , directory , clusterName);
	}
 }
 
 public static class StringArrayWritable extends ArrayWritable
 {
	 public StringArrayWritable()
	 {
		 super(Text.class);
	 }
	 
	 public String[] convert()
	 {
		 Writable[] writable = super.get();
		 String[] d = new String[writable.length];
		 for(int i = 0; i < writable.length; i++)
		 {
			 Text text  = (Text) writable[i];
			 d[i] = text.toString();
		 }
		 return d;
	 }
	 
	 public StringArrayWritable(String[] strings) {
            super(Text.class);
            Text[] texts = new Text[strings.length];
            for (int i = 0; i < strings.length; i++) {
                texts[i] = new Text(strings[i]);
            }
            set(texts);
        }
 }
 
 //calculation & update
  public static class KMeansMap extends Mapper<LongWritable, Text, IntWritable, StringArrayWritable> {
	
	 private int iterate = 0;
	 private int clusterNum = 0;
	 private double[][] centroid;
	 private String directory = "KMeans";
	//讀出分群數、輸出資料夾及群心
	protected void setup(Context context) throws IOException, InterruptedException {
        Configuration conf = context.getConfiguration();
        iterate = Integer.parseInt(conf.get("Iterate"));
		clusterNum =  Integer.parseInt(conf.get("clusterNum"));
		directory = conf.get("Directory");
		centroid = readCentroid(clusterNum , iterate , directory);
    }
        
	//calculate distance & cluster
    public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
		String line = value.toString();
		String[] data = line.split(",");
		double[] point = new double[4];
		int i = 0;
		try{
			data[1] = data[1].replaceAll("\\s+"," ");
			for(i = 1; i < 5; i++)
			{
				point[i - 1] = Double.parseDouble(data[i]);
			}
		}catch(Exception e){
			point[i - 1] = 0;
		}
		int clusterId = 0;
		double min = 0;
		double sum;
		//計算各點與群心之距離，並找出相差最短之群心為一群
		for(i = 0; i < centroid.length; i++)
		{
			sum = 0;
			for(int j = 0; j < 4; j++)
			{
				sum = sum + (centroid[i][j] - point[j]) * (centroid[i][j] - point[j]);
			}
			if(min == 0)
			{
				min = sum;
				clusterId = i + 1;
			}
			else
			{
				if(sum < min)
				{
					min = sum;
					clusterId = i + 1;
				}
			}
		}
		
		context.write(new IntWritable(clusterId) , new StringArrayWritable(Arrays.copyOfRange(data , 1 , data.length)));
    }
 } 
        
 public static class KMeansReduce extends Reducer<IntWritable, StringArrayWritable, Text, Text> {
	
	private int iterate = 0;
	private double[][] centroid ;
	 private int clusterNum;
	 private String directory = "KMeans";
	 private String[] clusterName;
	 private String lastPoint;
	//讀出分群數、輸出資料夾
	protected void setup(Context context) throws IOException, InterruptedException {
        Configuration conf = context.getConfiguration();

        iterate = Integer.parseInt(conf.get("Iterate"));
		clusterNum =  Integer.parseInt(conf.get("clusterNum"));
		centroid = new double[clusterNum][4];
		directory = conf.get("Directory");
		clusterName = new String[clusterNum];
		for(int i = 0; i < clusterName.length; i++)
		 {
			 clusterName[i] = "";
		 }
		lastPoint = conf.get("lastPoint");
    }
	//各群計算各自群心
    public void reduce(IntWritable key, Iterable<StringArrayWritable> values, Context context) 
      throws IOException, InterruptedException {
		double[] center = new double[4];
		 int valSize = 0;
		 int k = key.get() - 1;
		 Iterator<StringArrayWritable> it = values.iterator();
		 if(!it.hasNext())
		 {
			 //該群若無資料點，則選擇一資料點作為群心
			 String[] d = lastPoint.split(",");
			 for(int i = 0; i < d.length; i++)
			{
				centroid[k][i] = Double.parseDouble(d[i]);
			}
			 context.write(new Text(key.toString() + ",") , new Text(lastPoint));
		 }
		 else
		 {
			for(int i = 0; i < center.length; i++)
			 {
				 center[i] = 0;
			 }
			 //將分群解果輸出至檔案
			for (StringArrayWritable val : values) {
				String[] point = val.toStrings();
				if(!clusterName[k].contains(point[4]))
				{
					clusterName[k] = clusterName[k] + point[4] + "&";
				}
				valSize ++;
				String data = "";
				for(int i = 0; i < point.length - 1; i++)
				{
					try{
						data = data +  point[i] + ",";
						center[i] = center[i] + Double.parseDouble(point[i]);
					}catch(Exception e){
						System.out.println("error data");
					}
				}
				lastPoint = data + point[4];
				context.write(new Text(key.toString() + ",") , new Text(data + point[4]));
			}
				//更新群心
			for(int i = 0; i < center.length; i++)
			{
				center[i] = center[i] / valSize;
				centroid[k][i] = center[i];
				System.out.print(center[i]);
			}
		 }
    }
	//將更新的群心輸出至檔案
	protected void cleanup(Context context) throws IOException,InterruptedException
	{
		Configuration conf = context.getConfiguration();
		conf.set("lastPoint" , lastPoint );
		updateCentroid(centroid , iterate , directory , clusterName);
	}
 }
 
 
        
 public static void main(String[] args) throws Exception {
	String clusterNum = "1";
	String directory = "KMeans";
	double th = 1;
	int iterate = 0;
	//檢查輸入之參數輸出檔案、分群數、閥值
	if(args.length != 4)
	{
		clusterNum = args[2];
		directory = args[1];
		th = Double.parseDouble(args[3]);
	}
	else
	{
		System.exit(0);
	}
	//初始分群結果及群心
	Configuration conf_init = new Configuration();
	conf_init.set("ClusteringNum" , clusterNum );
	conf_init.set("Directory" , directory);
    Job job_init = new Job(conf_init, "KMeans_init");
    job_init.setMapOutputKeyClass(IntWritable.class);
	job_init.setMapOutputValueClass(StringArrayWritable.class);
	
    job_init.setOutputKeyClass(Text.class);
    job_init.setOutputValueClass(Text.class);
        
    job_init.setMapperClass(InitMap.class);
    job_init.setReducerClass(InitReduce.class);
    job_init.setJarByClass(KMeansForIris.class);
        
    job_init.setInputFormatClass(TextInputFormat.class);
    job_init.setOutputFormatClass(TextOutputFormat.class);
        
    FileInputFormat.addInputPath(job_init, new Path(args[0]));
    FileOutputFormat.setOutputPath(job_init, new Path("/kmeans/" + directory + "/iterate" + Integer.toString(iterate)));
    job_init.waitForCompletion(true);
	String lastPoint = conf_init.get("lastPoint");
	//進行遞迴直到收斂
	Configuration conf = new Configuration();
	while(!validation(Integer.parseInt(clusterNum) , iterate , directory , th))
	{
		if((lastPoint = conf.get("lastPoint")) != null)
		{
			conf.set("lastPoint" , lastPoint);
		}
		iterate++;
		conf.set("Iterate" , Integer.toString(iterate));
		conf.set("clusterNum" , clusterNum);
		conf.set("Directory" , directory);
		Job job = new Job(conf, "KMeans");
		job.setMapOutputKeyClass(IntWritable.class);
		job.setMapOutputValueClass(StringArrayWritable.class);
		
		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(Text.class);
			
		job.setMapperClass(KMeansMap.class);
		job.setReducerClass(KMeansReduce.class);
		job.setJarByClass(KMeansForIris.class);
			
		job.setInputFormatClass(TextInputFormat.class);
		job.setOutputFormatClass(TextOutputFormat.class);
			
		FileInputFormat.addInputPath(job, new Path("/kmeans/" + directory + "/iterate" + Integer.toString(iterate - 1) + "/part-r-00000"));
		FileOutputFormat.setOutputPath(job, new Path("/kmeans/" + directory + "/iterate" + Integer.toString(iterate)));
		job.waitForCompletion(true);
	}
	
 }
    //驗證所有群心移動距離
	public static boolean validation( int clusterNum , int i ,String dir , double th)
	{
		if(i == 0)
		{
			return false;
		}
		else if(i == 1000)
		{
			return true;
		}
		double[][] centroid = readCentroid(clusterNum , i++ , dir);
		double[][] centroid_old = readCentroid(clusterNum , i , dir);
		if(centroid == null || centroid_old == null)
		{
			System.out.println("read File error");
			return true;
		}
		
		for(int k = 0; k < clusterNum; k++)
		{
			double diff;
			for(int j = 0; j < centroid[k].length;j++)
			{
				diff = 0;
				System.out.println(centroid[k][j]);
				System.out.println(centroid_old[k][j]);
				diff = (centroid[k][j] - centroid_old[k][j]) * (centroid[k][j] - centroid_old[k][j]);
				System.out.println(diff);
				System.out.println(th);
				if(diff > th)
					return false;
			}

		}
		return true;
	}
	
	public static void updateCentroid(double[][] c, int i , String dir , String[] clusterName)
	{
		try{
			Configuration conf = new Configuration();
			FileSystem hdfs = FileSystem.get(conf);
			Path centroidFile = new Path("/kmeans/" + dir + "/centroid" + Integer.toString(i));
			OutputStream out;

			out = hdfs.create(centroidFile);
			OutputStreamWriter osw = new OutputStreamWriter( out);
			for(int j = 0 ;j < c.length; j++)
			{
				for(int k = 0; k < c[j].length; k++)
				{
					osw.append(Double.toString(c[j][k]) + ",");
					System.out.print(c[j] + ",");
				}
				osw.append(clusterName[j]);
				osw.append("\n");
			}

			System.out.print("\n");
			osw.flush();
			osw.close();
		}catch(Exception e)
		{
			
			System.out.println("updateCentroid error");
		}
		
	}
	
	public static double[][] readCentroid(int CN , int i , String dir)
	{
		double[][] centroid = new double[CN][4];
		try{
			Configuration conf = new Configuration();
			FileSystem hdfs = FileSystem.get(conf);
			InputStreamReader is = new InputStreamReader(hdfs.open(new Path("/kmeans/" + dir + "/centroid" + Integer.toString(i - 1))));
			BufferedReader br=new BufferedReader(is);
			String line = br.readLine();
			int c = 0;
			while(line != null)
			{
				String[] val = line.split(",");
				for(int j = 0; j < val.length - 1; j++)
				{
					centroid[c][j] = Double.parseDouble(val[j]);
				}
				System.out.println(line);
				line = br.readLine();
				c++;
			}
			br.close();
		}catch(Exception e){
			System.out.println("read File error");
			return null;
		}
		return centroid;
	}
 
        
}

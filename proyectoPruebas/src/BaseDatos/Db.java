package BaseDatos;

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;

import Scraper.LlamadaNoticias;
import datos.Noticia;
import datos.Usuario;



public class Db {
	
	static LlamadaNoticias oli = new LlamadaNoticias();
	static Statement statement;
	static String noticia,link;
	static String nombre,pass;
	
	public static Connection iniDB(String url){
		
		   try {
			Class.forName("org.sqlite.JDBC"); 
			Connection connection =  DriverManager.getConnection("jdbc:sqlite:"+url);
			return connection;
		} catch (ClassNotFoundException | SQLException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
			return null;
		}
		    
          
}
	public static void CrearTablas(String url){
		Connection con = iniDB(url);
		
		try {
			statement = con.createStatement();
			
			 statement.executeUpdate("create table if not exists noticias (titulo text UNIQUE, link text,categoria text,fecha date)");
		     statement.executeUpdate("create table if not exists usuario (usuario text UNIQUE, pass text)");
		} catch (SQLException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}finally {

		      try

		      {

		        if(con != null)

		          con.close();

		      }

		      catch(SQLException e)

		      {

		        // Cierre de conexión fallido

		        System.err.println(e);

		      }

		    }
	}
	public static void insertNews(String url,ArrayList<Noticia> arr){
		String date= new SimpleDateFormat("yyyy-MM-dd").format(new Date());
		Connection con= iniDB(url);
		String sql = "insert into noticias values (?,?,?,?);";
		PreparedStatement sta = null;
		try {
			sta = con.prepareStatement(sql);
			con.setAutoCommit(false);
			for(Noticia not : arr) {
				sta.setString(1, not.getTitulo());
				sta.setString(2, not.getLink());
				sta.setString(3, not.getCategoria());
				sta.setDate(4, java.sql.Date.valueOf(date));
				sta.addBatch();
			}
			sta.executeBatch();
			con.commit();
			
		} catch (SQLException e) {
			try{
				con.rollback();
			}catch (Exception e1) {
				// TODO: handle exception
				e1.printStackTrace();
			}
		
			}finally {

			      try

			      {

			        if(con != null)

			          con.close();

			      }

			      catch(SQLException e)

			      {

			        // Cierre de conexión fallido

			        System.err.println(e);

			      }
			      }

	}
	public static ArrayList<Noticia> extraerNoticia(String url,String cat){
		Connection con = iniDB(url);
		ResultSet rs = null;
		PreparedStatement pr = null;
		ArrayList<Noticia> array = new ArrayList<Noticia>();
		String sql="select titulo,link,categoria,fecha from noticias WHERE categoria='"+cat+"'";
		try {
			pr = con.prepareStatement(sql);
			rs = pr.executeQuery();
			

			while(rs.next()) {
				array.add(new Noticia(rs.getString(1),
						rs.getString(2),
						rs.getString(3),
						new SimpleDateFormat("yyyy-MM-dd").format(rs.getDate(4))));
			}
			return array;
		} catch(SQLException e) {
			e.printStackTrace();
			return null;
		} finally {

		      try

		      {

		        if(con != null)

		          con.close();

		      }

		      catch(SQLException e)

		      {

		        // Cierre de conexión fallido

		        System.err.println(e);

		      }
		      }
		
	}
	public static void ComprobarUsuario(String url,Usuario us){
		Connection con = iniDB(url);
		PreparedStatement pr = null;
		String usu = us.getUser();
		String pass = us.getPass();
		ResultSet rs = null;
		String sql = "select * from usuario where usuario = '"+usu+"' ";
		try {
			pr = con.prepareStatement(sql);
			rs = pr.executeQuery();
			while(rs.next()){
				if(rs.getString("usuario").equals(usu)||rs.getString("pass").equals(pass)){
					System.out.println("Bienvenido al sistema");
				}else{
					System.out.println("Erro");
				}
				}
			}
					
				

			
		 catch (SQLException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
	}
	
    
    public static void CrearUsuario(Usuario us,String url){
    	
    	Connection con = iniDB(url);
    	PreparedStatement pr = null;
    	String sql = "insert into usuario values (?,?);";
    	
    	try {
    		
			pr = con.prepareStatement(sql);
			con.setAutoCommit(true);
			pr.setString(1, us.getUser());
			pr.setString(2, us.getPass());
		} catch (SQLException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}finally {

		      try

		      {

		        if(con != null)

		          con.close();

		      }

		      catch(SQLException e)

		      {

		        // Cierre de conexión fallido

		        System.err.println(e);

		      }
		      }
    	
    }

	
	  
}

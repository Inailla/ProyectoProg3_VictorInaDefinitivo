package Scraper;

import java.io.IOException;
import java.util.ArrayList;

import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.jsoup.select.Elements;

import BaseDatos.Db;
import datos.Noticia;

public class LlamadaNoticias {
	private String dir = System.getProperty("user.dir");
	private String url;

	public LlamadaNoticias() {
		url = dir + "/" + "sample1.db";
		Db.CrearTablas(url);
	}
	
	public ArrayList<Noticia> noticasEco() throws IOException{
		
		ArrayList<Noticia> Eco = new ArrayList<>();
		String urlEco = "http://www.eleconomista.es/rss/rss-economia.php";
		Document d2 = Jsoup.connect(urlEco).get();
		Elements el2 = d2.select("title");
		Elements el3 = d2.select("link");
		
		for(int i = 0; i < el2.size(); i++) {
			Eco.add(new Noticia(el2.get(i).text(), el3.get(i).text()));
		}
		Db.insertNews(url, Eco, "Economia");
		return Eco;
			
	}
	
	public ArrayList<Noticia> noticiasDepor() throws IOException{
		
		ArrayList<Noticia> link = new ArrayList<>();
		String urlDeportes = "http://www.losotros18.com/la-liga/";
		Document d = Jsoup.connect(urlDeportes).get();
		Elements el = d.select("div#wrapper");
		for (org.jsoup.nodes.Element element : el.select("div.post-thumbnail")) {
			String title = element.select("div.post-thumbnail a").attr("title");
			String url2 = element.select("div.post-thumbnail a").attr("href");
			link.add(new Noticia(title, url2));
			
			}
		   Db.insertNews(url, link, "Deportes");		
             return link;
	}
    public ArrayList<Noticia> noticasUlt() throws IOException{
		
		ArrayList<Noticia> ult = new ArrayList<>();
		String urlEco = "http://ep00.epimg.net/rss/tags/ultimas_noticias.xml";
		Document d2 = Jsoup.connect(urlEco).get();
		Elements el2 = d2.select("title");
		Elements el3 = d2.select("link");
		
		for(int i = 0; i < el2.size(); i++) {
			ult.add(new Noticia(el2.get(i).text(), el3.get(i).text()));
		}
		Db.insertNews(url, ult, "ultimas");
		return ult;
			
	}
	
		
	
	}
	

package ahorcado.general;

public class Mensaje {

private String mensaje;
	
	public void muchasVidas (){
		mensaje="No te apresures aun te quedan varios intentos";
	}
	
	public void mitadVidas (){
		mensaje="Concentrate se te acaban las vidas";
	}
	
	public void pocasVidas (){
		mensaje="Ya no puedes fallar";
	}
	
	public void ganastes(){
		mensaje="Has ganado";
	}
	
	public void perdistes(){
		mensaje="Has perdido";
	}

	public String getMensaje() {
		return mensaje;
	}
	
	
}

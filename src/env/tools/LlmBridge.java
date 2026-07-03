import java.io.BufferedReader;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.URL;
import java.net.URLEncoder;
import java.util.HashMap;
import java.util.Map;

import cartago.Artifact;
import cartago.OPERATION;

/**
 * Artifact that implements the auction.
 */
public class LlmBridge extends Artifact {
	
	/**
	 * url of the python server
	 */
	public static String	SERVER_URL;
	/**
	 * port of the python server
	 */
	public static int		SERVER_PORT;
	/**
	 * endpoint for the server route
	 */
	public static String	SOLVE_SERVICE;
	/**
	 * parameter for the input data
	 */
	public static String	INPUT_DATA_PARAM;
	
	{ // use the same block of constants from server.py
		SERVER_URL = "http://localhost";
		SERVER_PORT = 5565;
		SOLVE_SERVICE = "solve";
		INPUT_DATA_PARAM = "input_data";
	}
	
	void init() {
		// optional setup logic when the artifact is created
		log("LLM Bridge artifact up.");
	}
	
	@OPERATION
	void solve() {
		try {
			System.out.println("Solving...");
			// Create the request data
			Map<String, String> postData = new HashMap<>();
			postData.put(INPUT_DATA_PARAM, "test");
			
			// Set up the connection
			HttpURLConnection connection = setupConnection(SOLVE_SERVICE, "POST", postData);
			// Check the response
			String response = checkResponse(connection);
			if(response != null) {
				System.out.println("Response: " + response);
			}
		} catch(Exception e) {
			System.out.println("Exception: " + e.getMessage());
		}
	}
	
	protected static HttpURLConnection setupConnection(String route_endpoint, String request_method,
			Map<String, String> params) {
		try {
			String location = SERVER_URL + ":" + SERVER_PORT + "/" + route_endpoint;
			URL url = new URL(location);
			HttpURLConnection connection = (HttpURLConnection) url.openConnection();
			connection.setRequestMethod(request_method);
			connection.setRequestProperty("Content-Type", "application/x-www-form-urlencoded");
			connection.setDoOutput(true);
			
			if(params != null) {
				String PostData = "";
				for(Map.Entry<String, String> param : params.entrySet()) {
					PostData += param.getKey() + "=" + URLEncoder.encode(param.getValue(), "UTF-8") + "&";
				}
				DataOutputStream wr = new DataOutputStream(connection.getOutputStream());
				wr.writeBytes(PostData);
				wr.flush();
			}
			return connection;
			
		} catch(IOException e) {
			System.out.println("Error: " + e.getMessage() + " | ");
			e.printStackTrace();
			return null;
		}
	}
	
	/**
	 * Method to check the response from the server. If the response code returned by the server is OK, a string
	 * containing the response is returned
	 *
	 * @param connection
	 *            The connection to the server
	 * 			
	 * @return The response from the server, if the response code is OK. Null otherwise
	 */
	protected static String checkResponse(HttpURLConnection connection) {
		if(connection == null)
			return null;
		try {
			String response = "";
			int responseCode = connection.getResponseCode();
			boolean iserror = responseCode >= 400;
			try (BufferedReader in = new BufferedReader(
					new InputStreamReader(!iserror ? connection.getInputStream() : connection.getErrorStream()))) {
				String line;
				while((line = in.readLine()) != null) {
					response += line;
				}
				if(iserror)
					System.out.println("Error:" + responseCode + "|" + connection.getResponseMessage() + ". Response: "
							+ response);
				else
					System.out.println("Response: " + response);
				return !iserror ? response : null;
			}
			
		} catch(IOException e) {
			System.out.println("Error: " + e.getMessage() + " | ");
			e.printStackTrace();
			return null;
		}
	}
}

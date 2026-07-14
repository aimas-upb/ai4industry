import java.io.BufferedReader;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.StringReader;
import java.net.HttpURLConnection;
import java.net.URL;
import java.net.URLEncoder;
import java.util.HashMap;
import java.util.Map;
import java.util.Timer;
import java.util.TimerTask;

import javax.json.Json;
import javax.json.JsonObject;
import javax.json.JsonReader;

import cartago.Artifact;
import cartago.INTERNAL_OPERATION;
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
	 * endpoint for the server route -- the solve service
	 */
	public static String	SOLVE_SERVICE;
	/**
	 * endpoint for the server route -- getting the status
	 */
	public static String	STATUS_SERVICE;
	/**
	 * parameter for the input data
	 */
	public static String	INPUT_DATA_PARAM;
	
	{ // use the same block of constants from server.py
		SERVER_URL = "http://localhost";
		SERVER_PORT = 5565;
		SOLVE_SERVICE = "solve";
		STATUS_SERVICE = "status";
		INPUT_DATA_PARAM = "input_data";
	}
	
	public static String RESULT_PROPERTY_NAME = "llmResult";
	protected static final long	CHECK_PERIOD_MS	= 2000L;
	protected Timer				timer;
	protected static int N_RETRIES = 3;
	protected int retries_left = 0;
	protected String currentRequestUri = null; // URI of the current request status resource
	
	void init() {
		// optional setup logic when the artifact is created
		log("LLM Bridge artifact up.");
		defineObsProperty(RESULT_PROPERTY_NAME, "");
	}
	
	@OPERATION
	void solve(String goal) {
		try {
			System.out.println("Solving...");
			// Create the request data
			Map<String, String> postData = new HashMap<>();
			postData.put(INPUT_DATA_PARAM, goal);

			// Set up the connection
			HttpURLConnection connection = setupConnection(SOLVE_SERVICE, "POST", postData);
			// Check the response (expects 202 with request_uri in JSON)
			String response = checkResponseWithCode(connection);
			if(response != null) {
				System.out.println("Response: OK. Starting timer.");
				// Parse request_uri from JSON response: {"request_uri": "...", ...}
				currentRequestUri = extractRequestUri(response);
				if(currentRequestUri != null) {
					System.out.println("Request URI: " + currentRequestUri);
					getObsProperty(RESULT_PROPERTY_NAME).updateValue("");
					retries_left = N_RETRIES;
					startCheckTimer();
				} else {
					System.out.println("Error: Could not extract request_uri from response");
				}
			}
			else
				System.out.println("Response: Error");
		} catch(Exception e) {
			System.out.println("Exception: " + e.getMessage());
		}
	}
	
	protected void startCheckTimer() {
		stopCheckTimer(); // avoid duplicate timers if solve() is called again
		
		timer = new Timer(true); // daemon thread
		timer.scheduleAtFixedRate(new TimerTask() {
			@Override
			public void run() {
				// hop back onto the artifact's own thread to touch state safely
				execInternalOp("checkResult");
			}
		}, CHECK_PERIOD_MS, CHECK_PERIOD_MS);
	}
	
	private void stopCheckTimer() {
		if(timer != null) {
			timer.cancel();
			timer = null;
		}
	}
	
	@INTERNAL_OPERATION
	void checkResult() {
		try {
			if(currentRequestUri == null) {
				System.out.println("Error: No current request URI. Stopping timer.");
				stopCheckTimer();
				return;
			}

			System.out.println("Checking response...");
			// GET the status resource using the request URI
			URL url = new URL(currentRequestUri);
			HttpURLConnection connection = (HttpURLConnection) url.openConnection();
			connection.setRequestMethod("GET");
			connection.setRequestProperty("Content-Type", "application/json");
			connection.setDoOutput(false);

			// Check the response
			String response = checkResponse(connection);
			if(response != null) {
				System.out.println("Response received.");
				// Extract execution_result status from the response JSON
				String status = extractExecutionResult(response);
				if(status != null && (status.equals("SUCCESS") || status.equals("FAILURE") || status.equals("TIMEOUT"))) {
					// Terminal status reached - update property and stop polling
					getObsProperty(RESULT_PROPERTY_NAME).updateValue(response);
					stopCheckTimer();
				} else {
					// Still running or just received initial response - keep polling
					System.out.println("Request still " + (status != null ? status : "processing"));
				}
			}
			else {
				System.out.println("Response: Error");
				retries_left--;
				if(retries_left <= 0) {
					System.out.println("Stopped waiting for result.");
					stopCheckTimer();
				}
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
			boolean isOK = responseCode == 200;
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
				return !iserror && isOK ? response : null;
			}

		} catch(IOException e) {
			System.out.println("Error: " + e.getMessage() + " | ");
			e.printStackTrace();
			return null;
		}
	}

	/**
	 * Check response and return the raw response body regardless of HTTP status code.
	 * Used for parsing 202 Accepted responses which contain JSON with request_uri.
	 *
	 * @param connection The connection to the server
	 * @return The response body, or null on error
	 */
	protected static String checkResponseWithCode(HttpURLConnection connection) {
		if(connection == null)
			return null;
		try {
			String response = "";
			int responseCode = connection.getResponseCode();
			boolean iserror = responseCode >= 400;
			boolean isAccepted = responseCode == 202;
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
				return (!iserror && isAccepted) ? response : null;
			}

		} catch(IOException e) {
			System.out.println("Error: " + e.getMessage() + " | ");
			e.printStackTrace();
			return null;
		}
	}

	/**
	 * Extract request_uri from JSON response.
	 *
	 * @param jsonResponse The JSON response string
	 * @return The request_uri value, or null if not found or parsing fails
	 */
	protected static String extractRequestUri(String jsonResponse) {
		if(jsonResponse == null || jsonResponse.isEmpty())
			return null;

		try (JsonReader reader = Json.createReader(new StringReader(jsonResponse))) {
			JsonObject obj = reader.readObject();
			if(obj.containsKey("request_uri")) {
				return obj.getString("request_uri");
			}
		} catch(Exception e) {
			System.out.println("Error parsing request_uri from JSON: " + e.getMessage());
		}
		return null;
	}

	/**
	 * Extract execution_result status from JSON response.
	 *
	 * @param jsonResponse The JSON response string
	 * @return The execution_result status (RUNNING/SUCCESS/FAILURE/TIMEOUT), or null if not found
	 */
	protected static String extractExecutionResult(String jsonResponse) {
		if(jsonResponse == null || jsonResponse.isEmpty())
			return null;

		try (JsonReader reader = Json.createReader(new StringReader(jsonResponse))) {
			JsonObject obj = reader.readObject();
			if(obj.containsKey("execution_result")) {
				return obj.getString("execution_result");
			}
		} catch(Exception e) {
			System.out.println("Error parsing execution_result from JSON: " + e.getMessage());
		}
		return null;
	}
}

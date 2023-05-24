//
//  AuthService.swift
//  FoodAssistant
//
//  Created by Elton Miao on 28/12/2022.
//

import Foundation
import Combine

class AuthService: ObservableObject {
    @Published var authenticated: Bool = false
    @Published var token: String? = nil
    @Published var user: User? = nil
    @Published var avator: URL? = nil
    
    var scheme: String
    var host: String
    var port: Int
    
    var subscriptions = Set<AnyCancellable>()
    
    init (scheme: String, host: String, port: Int){
        self.scheme = scheme
        self.host = host
        self.port = port
        
        guard let savedToken: String = UserDefaults
            .standard
            .string(forKey: "token") else {
                self.authenticated = false
                return
            }
        
        self.authenticated = true
        self.token = savedToken
    }
    
    func getUserProfile() {
        AppState
            .shared
            .dataService
            .get(type: User.self, path: "/api/userprofile/")
            .receive(on: DispatchQueue.main)
            .sink { (completion: Subscribers.Completion<Error>) in
                switch completion {
                case let .failure(error):
                    print("Couldn't get users: \(error)")
                case .finished: break
                }
            } receiveValue: { (user: User) in //[weak self]
                self.user = user
                self.avator = user.userAvator
            }
            .store(in: &subscriptions)
    }
    
    func uploadUserProfile(imageData: Data) {
        AppState
            .shared
            .dataService
            .putFile(type: User.self, path: "/upload/\(UUID().uuidString).jpg", data: imageData)
            .receive(on: DispatchQueue.main)
            .sink(receiveCompletion: { (completion) in
                switch completion {
                case let .failure(error):
                    print("Couldn't get users: \(error)")
                case .finished: break
                }
            }) {  user in //[weak self]
                
                self.user = user
                self.avator = user.userAvator
                //                    print(self.user)
            }
            .store(in: &subscriptions)
        
    }
    
    func register(
        userName: String,
        password: String,
        lastname: String,
        firstname: String,
        genderIndex: Int,
        birthDate: Date,
        imageData: Data?
    ) {
        AppState
            .shared
            .dataService
            .post(
                object: User(
                    email: userName,
                    password: password,
                    lastName: lastname,
                    firstName: firstname,
                    dateOfBirth: birthDate,
                    gender: genderIndex == 0 ? "M" : "F"
                ),
                type: User.self,
                path: "/api/users/create/"
            )
            .receive(on: DispatchQueue.main)
            .sink { (completion: Subscribers.Completion<Error>) in
                switch completion {
                case let .failure(error):
                    print("Couldn't get userssssss: \(error)")
                    //                self.login(userName: userName, password: password)
                    //                AppState.shared.authService.upload_user_profile(image_data: imageData!)
                case .finished:
                    break
                    //                self.login(userName: userName, password: password)
                    //                AppState.shared.authService.upload_user_profile(image_data: imageData!)
                }
            } receiveValue: { (user: User) in //[weak self]
                //            self.login(userName: userName, password: password)
                self.user = user
                //print(self.user)
            }
            .store(in: &subscriptions)
        
    }
    
    func login(userName: String, password: String) {
        print("Login")
        var components = URLComponents()
        components.scheme = self.scheme
        components.host = self.host
        components.port = self.port
        components.path = "/api/token/"
        
        guard let url = components.url else {
            preconditionFailure("Invalid URL components: \(components)")
        }
        var urlRequest = URLRequest(url: url, cachePolicy: .reloadIgnoringLocalAndRemoteCacheData)
        urlRequest.httpMethod = "POST"
        let parameters: [String: Any] = [
            "email": userName,
            "password": password
        ]
        urlRequest.addValue("application/json", forHTTPHeaderField: "Content-Type")
        do {
            urlRequest.httpBody = try JSONSerialization.data(withJSONObject: parameters, options: .prettyPrinted)
        } catch let error {
            print(error.localizedDescription)
            return
        }
        
        let task = URLSession.shared.dataTask(with: urlRequest) { data, response, error in
            
            if let error = error {
                print("Post Request Error: \(error.localizedDescription)")
                return
            }
            
            // ensure there is valid response code returned from this HTTP response
            guard let httpResponse = response as? HTTPURLResponse,
                  (200...299).contains(httpResponse.statusCode)
            else {
                print("Invalid Response received from the server")
                return
            }
            
            // ensure there is data returned
            guard let responseData = data else {
                print("nil Data received from the server")
                return
            }
            
            do {
                // create json object from data or use JSONDecoder to convert to Model stuct
                if let jsonResponse = try JSONSerialization.jsonObject(with: responseData, options: .mutableContainers) as? [String: String] {
                    print(jsonResponse)
                    DispatchQueue.main.async {
                        self.token = jsonResponse["access"]
                        UserDefaults.standard.set(self.token, forKey: "token")
                        self.authenticated = true
                    }
                    
                    // handle json response
                } else {
                    print("data maybe corrupted or in wrong format")
                    throw URLError(.badServerResponse)
                }
            } catch let error {
                print(error.localizedDescription)
            }
        }
        task.resume()
    }
    
    func logout() {
        print("logout")
        self.token = nil
        UserDefaults.standard.removeObject(forKey: "token")
        //        self.keychain["token"] = self.token
        self.authenticated = false
    }
    
}



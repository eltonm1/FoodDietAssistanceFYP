//
//  APIService.swift
//  FoodAssistant
//
//  Created by Elton Miao on 28/12/2022.
//

import Foundation
import Combine
import UIKit

class APIService {
    
    var scheme: String
    var host: String
    var port: Int
    var authService: AuthService
    private var jsonDecoder: JSONDecoder
    
    init(
        scheme: String,
        host: String,
        port: Int,
        authService: AuthService
    ) {
        self.scheme = scheme
        self.host = host
        self.port = port
        self.authService = authService
        
        self.jsonDecoder = JSONDecoder()
        let dateFormatter = DateFormatter()
        dateFormatter.dateFormat = "yyyy-MM-dd'T'HH:mm:ssZ"
        self.jsonDecoder.dateDecodingStrategy = .formatted(dateFormatter)
    }
    
    // Get Decodable Data from designated path
    func get<T: Decodable>(type: T.Type, path: String) -> AnyPublisher<T, Error>{
        let urlRequest: URLRequest = makeURLRequest(from: path)
        
        let dataTaskPublisher: AnyPublisher<T, Error> = URLSession
            .shared
            .dataTaskPublisher(for: urlRequest)
            .map { (data: Data, response: URLResponse) -> Data in data }
            .tryCatch { (failure: URLSession.DataTaskPublisher.Failure) -> Just<Data> in
                //Try To Find From Cache
                guard let cacheedResponse: CachedURLResponse = URLCache
                    .shared
                    .cachedResponse(for: urlRequest) else {
                    throw failure
                }
                return Just(cacheedResponse.data)
            }
            .decode(type: T.self, decoder: self.jsonDecoder)
            .eraseToAnyPublisher()
        
        return dataTaskPublisher
    }

    /// Get the item of specified type in an asynchronous manner
    ///
    /// - Parameters:
    ///   - type: The type of the item to be retrieved
    ///   - path: The path of the item to be retrieved
    /// - Returns: The decoded object of type T
    func get<T: Decodable>(type: T.Type, path: String) async throws -> T {
        let urlRequest: URLRequest = makeURLRequest(from: path)
        return try await performCall(urlRequest: urlRequest)
    }
    
    func post<T: Codable>(type: T.Type, image: UIImage, host: String, port: Int, path: String) async throws -> T {
        var components = URLComponents()
        components.scheme = self.scheme
        components.host = host
        components.port = port
        components.path = path //"/api/xxx/"
        
        guard let url: URL = components.url else {
            preconditionFailure("Invalid URL components: \(components)")
        }
        var urlRequest = URLRequest(url: url, cachePolicy: .reloadIgnoringLocalCacheData)
        URLCache.shared.removeAllCachedResponses()
        urlRequest.httpMethod = "POST"
        urlRequest.timeoutInterval = 8.0
        
        let uuid: String = UUID().uuidString
        let CRLF: String = "\r\n"
        let filename: String = uuid + ".jpg"
        let type: String = "image/jpeg"
        let boundary = String(format: "----iOSURLSessionBoundary.%08x%08x", arc4random(), arc4random())
        
        var body: Data = Data()
        body.append(("--\(boundary)" + CRLF).data(using: .utf8)!)
        body.append("Content-Disposition: form-data; name=\"formName\"; filename=\"\(filename)\"\r\n".data(using: .utf8)!)
        body.append(("Content-Type: \(type)" + CRLF + CRLF).data(using: .utf8)!)
        
        if let imagedata: Data = image.jpegData(compressionQuality: 1) {
            body.append(imagedata)
        } else {
            print("NO JPG DATA")
        }
        body.append(CRLF.data(using: .utf8)!)

        // footer //
        body.append(("--\(boundary)--" + CRLF).data(using: .utf8)!)
        urlRequest.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
        urlRequest.httpBody = body
        
        return try await performCall(urlRequest: urlRequest)
    }
    
    func post<T: Codable>(object: T, type: T.Type, host: String, port: Int, path: String) async throws -> T {
        var components = URLComponents()
        components.scheme = self.scheme
        components.host = host
        components.port = port
        components.path = path //"/api/xxx/"
        
        guard let url: URL = components.url else {
            preconditionFailure("Invalid URL components: \(components)")
        }
        var urlRequest = URLRequest(url: url, cachePolicy: .reloadIgnoringLocalCacheData)
        URLCache.shared.removeAllCachedResponses()
        urlRequest.timeoutInterval = 5.0
        urlRequest.httpMethod = "POST"
        urlRequest.addValue("application/json", forHTTPHeaderField: "Content-Type")
        do {
            urlRequest.httpBody = try JSONEncoder().encode(object)
        } catch let error {
            print(error.localizedDescription)
        }
        
        return try await performCall(urlRequest: urlRequest)
    }
    
    func post<T: Codable>(object: T, type: T.Type, path: String) async throws -> T{
        var urlRequest: URLRequest = makeURLRequest(from: path)
        urlRequest.httpMethod = "POST"
        urlRequest.addValue("application/json", forHTTPHeaderField: "Content-Type")
        do {
            urlRequest.httpBody = try JSONEncoder().encode(object)
        } catch let error {
            print(error.localizedDescription)
        }
        
        return try await performCall(urlRequest: urlRequest)
    }
    
    func post<T: Codable>(object: T, type: T.Type, path: String) -> AnyPublisher<T, Error>{
        var urlRequest: URLRequest = makeURLRequest(from: path)
        urlRequest.httpMethod = "POST"
        urlRequest.addValue("application/json", forHTTPHeaderField: "Content-Type")
        do {
            urlRequest.httpBody = try JSONEncoder().encode(object)
        } catch let error {
            print(error.localizedDescription)
        }
        
        let dataTaskPublisher: AnyPublisher<T, Error> = URLSession
            .shared
            .dataTaskPublisher(for: urlRequest)
            .map { (data: Data, response: URLResponse) -> Data in data }
            .tryCatch { (failure: URLSession.DataTaskPublisher.Failure) -> Just<Data> in
                //Try To Find From Cache
                guard let cacheedResponse: CachedURLResponse = URLCache
                    .shared
                    .cachedResponse(for: urlRequest) else {
                    throw failure
                }
                return Just(cacheedResponse.data)
            }
            .decode(type: T.self, decoder: self.jsonDecoder)
            .eraseToAnyPublisher()
        
        return dataTaskPublisher
    }
    
    func put<T: Codable>(object: T, type: T.Type, path: String) async throws -> T {
        var urlRequest: URLRequest = makeURLRequest(from: path)
        urlRequest.httpMethod = "PUT"
        urlRequest.addValue("application/json", forHTTPHeaderField: "Content-Type")
        do {
            urlRequest.httpBody = try JSONEncoder().encode(object)
        } catch let error {
            print(error.localizedDescription)
            
        }
        
        return try await performCall(urlRequest: urlRequest)
    }
    
    func put<T: Codable>(object: T, type: T.Type, path: String) -> AnyPublisher<T, Error>{
        var urlRequest: URLRequest = makeURLRequest(from: path)
        urlRequest.httpMethod = "PUT"
        urlRequest.addValue("application/json", forHTTPHeaderField: "Content-Type")
        do {
            urlRequest.httpBody = try JSONEncoder().encode(object)
        } catch let error {
            print(error.localizedDescription)
            
        }
        
        let dataTaskPublisher: AnyPublisher<T, Error> = URLSession
            .shared.dataTaskPublisher(for: urlRequest)
            .map { (data: Data, response: URLResponse) -> Data in data }
            .tryCatch{ (failure: URLSession.DataTaskPublisher.Failure) -> Just<Data> in
                //Try To Find From Cache
                guard let cacheedResponse: CachedURLResponse = URLCache
                    .shared
                    .cachedResponse(for: urlRequest) else {
                    throw failure
                }
                return Just(cacheedResponse.data)
            }
            .decode(type: T.self, decoder: self.jsonDecoder)
            .eraseToAnyPublisher()
        
        return dataTaskPublisher
    }
    
    func putFile<T: Codable>(type: T.Type, path: String, data: Data) -> AnyPublisher<T, Error>{
        var request: URLRequest = makeURLRequest(from: path)
        request.httpMethod = "PUT"
        request.httpBody = data
        let dataTaskPublisher = URLSession.shared.dataTaskPublisher(for: request)
            .map { (data: Data, response: URLResponse) -> Data in data }
            .decode(type: T.self, decoder: self.jsonDecoder)
            .eraseToAnyPublisher()
        
        return dataTaskPublisher
    }

    /// Make a URLRequest from path, incorporating the designated scheme, host, and port,
    /// and the authorization token.
    /// 
    /// - Parameter path: A String representing the path.
    /// - Returns: The URLRequest object.
    private func makeURLRequest(from path: String) -> URLRequest {
        var components = URLComponents()
        components.scheme = self.scheme
        components.host = self.host
        components.port = self.port
        components.path = path
        
        guard let url: URL = components.url else {
            preconditionFailure("Invalid URL components: \(components)")
        }

        var request = URLRequest(url: url, cachePolicy: .reloadIgnoringLocalCacheData)
        URLCache.shared.removeAllCachedResponses()
        if let token: String = self.authService.token {
            request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }

        return request
    }
    
    private func performCall<T: Decodable>(urlRequest: URLRequest) async throws -> T {
        let dataToBeDecoded: Data
        do {
            let (data, _): (Data, _) = try await URLSession.shared.data(for: urlRequest)
            dataToBeDecoded = data
        } catch {
            guard let cacheedResponse: CachedURLResponse = URLCache
                .shared
                .cachedResponse(for: urlRequest) else {
                throw error
            }
            dataToBeDecoded = cacheedResponse.data
        }
        return try jsonDecoder.decode(T.self, from: dataToBeDecoded)
    }
}

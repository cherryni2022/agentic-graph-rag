"""
Supabase Client SDK 使用示例
提供了两种方式：
1. 使用 Supabase Python SDK（高级封装）
2. 使用 asyncpg（底层 PostgreSQL 连接）
"""

import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from supabase import create_client, Client
import asyncio

# Load environment variables
load_dotenv()


class SupabaseService:
    """
    Supabase SDK 客户端服务
    
    使用场景：
    - 需要使用 Supabase 的认证、存储、实时订阅等功能
    - 更简洁的 API 调用方式
    - 自动处理连接池和重试逻辑
    """
    
    def __init__(self) -> None:
        """
        初始化 Supabase 客户端
        
        需要的环境变量：
        - SUPABASE_URL: 你的 Supabase 项目 URL (https://xxx.supabase.co)
        - SUPABASE_KEY: 你的 Supabase anon/service_role key
        """
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_KEY")
        
        if not supabase_url or not supabase_key:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in environment variables")
        
        # 创建 Supabase 客户端（自动管理连接池）
        self.client: Client = create_client(supabase_url, supabase_key)
    
    # ==================== Session 管理 ====================
    
    def create_session(
        self,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        创建新会话
        
        Args:
            user_id: 用户 ID
            metadata: 会话元数据
        
        Returns:
            创建的会话记录
        """
        data = {
            "user_id": user_id,
            "metadata": metadata or {}
        }
        
        result = self.client.table("sessions").insert(data).execute()
        return result.data[0] if result.data else None
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        获取会话信息
        
        Args:
            session_id: 会话 UUID
        
        Returns:
            会话数据或 None
        """
        result = self.client.table("sessions")\
            .select("*")\
            .eq("id", session_id)\
            .single()\
            .execute()
        
        return result.data if result.data else None
    
    def update_session(
        self,
        session_id: str,
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        更新会话元数据
        
        Args:
            session_id: 会话 UUID
            metadata: 新的元数据（会合并到现有元数据）
        
        Returns:
            更新后的会话数据
        """
        # 先获取现有元数据
        session = self.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        # 合并元数据
        current_metadata = session.get("metadata", {})
        current_metadata.update(metadata)
        
        result = self.client.table("sessions")\
            .update({"metadata": current_metadata})\
            .eq("id", session_id)\
            .execute()
        
        return result.data[0] if result.data else None
    
    # ==================== Message 管理 ====================
    
    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        添加消息到会话
        
        Args:
            session_id: 会话 UUID
            role: 消息角色 (user/assistant/system)
            content: 消息内容
            metadata: 消息元数据
        
        Returns:
            创建的消息记录
        """
        data = {
            "session_id": session_id,
            "role": role,
            "content": content,
            "metadata": metadata or {}
        }
        
        result = self.client.table("messages").insert(data).execute()
        return result.data[0] if result.data else None
    
    def get_session_messages(
        self,
        session_id: str,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        获取会话的所有消息
        
        Args:
            session_id: 会话 UUID
            limit: 最大返回数量
        
        Returns:
            消息列表（按时间排序）
        """
        query = self.client.table("messages")\
            .select("*")\
            .eq("session_id", session_id)\
            .order("created_at")
        
        if limit:
            query = query.limit(limit)
        
        result = query.execute()
        return result.data if result.data else []
    
    # ==================== Document 管理 ====================
    
    def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """
        获取文档信息
        
        Args:
            document_id: 文档 UUID
        
        Returns:
            文档数据或 None
        """
        result = self.client.table("documents")\
            .select("*")\
            .eq("id", document_id)\
            .single()\
            .execute()
        
        return result.data if result.data else None
    
    def list_documents(
        self,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        列出所有文档
        
        Args:
            limit: 最大返回数量
            offset: 跳过的记录数
        
        Returns:
            文档列表
        """
        result = self.client.table("documents")\
            .select("*, chunks:chunks(count)")\
            .order("created_at", desc=True)\
            .range(offset, offset + limit - 1)\
            .execute()
        
        return result.data if result.data else []
    
    # ==================== Vector Search ====================
    
    def vector_search(
        self,
        embedding: List[float],
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        向量相似度搜索
        
        Args:
            embedding: 查询向量
            limit: 最大返回数量
        
        Returns:
            匹配的文档块列表
        
        注意：需要在 Supabase 中创建 match_chunks 函数
        """
        result = self.client.rpc(
            "match_chunks",
            {
                "query_embedding": embedding,
                "match_count": limit
            }
        ).execute()
        
        return result.data if result.data else []
    
    def hybrid_search(
        self,
        embedding: List[float],
        query_text: str,
        limit: int = 10,
        text_weight: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        混合搜索（向量 + 关键词）
        
        Args:
            embedding: 查询向量
            query_text: 查询文本
            limit: 最大返回数量
            text_weight: 文本相似度权重 (0-1)
        
        Returns:
            匹配的文档块列表
        
        注意：需要在 Supabase 中创建 hybrid_search 函数
        """
        result = self.client.rpc(
            "hybrid_search",
            {
                "query_embedding": embedding,
                "query_text": query_text,
                "match_count": limit,
                "text_weight": text_weight
            }
        ).execute()
        
        return result.data if result.data else []
    
    # ==================== Storage 管理 ====================
    
    def upload_file(
        self,
        bucket_name: str,
        file_path: str,
        file_data: bytes,
        content_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        上传文件到 Supabase Storage
        
        Args:
            bucket_name: 存储桶名称
            file_path: 文件路径（在存储桶中的路径）
            file_data: 文件二进制数据
            content_type: 文件 MIME 类型
        
        Returns:
            上传结果
        """
        result = self.client.storage.from_(bucket_name).upload(
            file_path,
            file_data,
            {"content-type": content_type} if content_type else None
        )
        
        return result
    
    def download_file(
        self,
        bucket_name: str,
        file_path: str
    ) -> bytes:
        """
        从 Supabase Storage 下载文件
        
        Args:
            bucket_name: 存储桶名称
            file_path: 文件路径
        
        Returns:
            文件二进制数据
        """
        result = self.client.storage.from_(bucket_name).download(file_path)
        return result
    
    def get_public_url(
        self,
        bucket_name: str,
        file_path: str
    ) -> str:
        """
        获取文件的公开 URL
        
        Args:
            bucket_name: 存储桶名称
            file_path: 文件路径
        
        Returns:
            公开访问 URL
        """
        result = self.client.storage.from_(bucket_name).get_public_url(file_path)
        return result


# ==================== 使用示例 ====================

async def example_usage() -> None:
    """Supabase SDK 使用示例"""
    
    # 初始化客户端
    supabase = SupabaseService()
    
    # 1. 创建会话
    print("1. 创建会话...")
    session = supabase.create_session(
        user_id="user_123",
        metadata={"source": "web", "language": "zh-CN"}
    )
    print(f"   会话 ID: {session['id']}")
    
    # 2. 添加消息
    print("\n2. 添加消息...")
    message1 = supabase.add_message(
        session_id=session['id'],
        role="user",
        content="你好，请介绍一下 Supabase"
    )
    print(f"   消息 ID: {message1['id']}")
    
    message2 = supabase.add_message(
        session_id=session['id'],
        role="assistant",
        content="Supabase 是一个开源的 Firebase 替代品..."
    )
    
    # 3. 获取会话消息
    print("\n3. 获取会话消息...")
    messages = supabase.get_session_messages(session['id'])
    for msg in messages:
        print(f"   [{msg['role']}]: {msg['content'][:50]}...")
    
    # 4. 列出文档
    print("\n4. 列出文档...")
    documents = supabase.list_documents(limit=5)
    print(f"   找到 {len(documents)} 个文档")
    
    # 5. 向量搜索示例（需要先有 embedding）
    # query_embedding = [0.1, 0.2, 0.3, ...]  # 实际的向量
    # results = supabase.vector_search(query_embedding, limit=5)
    # print(f"   找到 {len(results)} 个相似结果")


if __name__ == "__main__":
    # 运行示例
    asyncio.run(example_usage())

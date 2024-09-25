from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from langchain_core.utils import get_from_dict_or_env
from pydantic import BaseModel

logger = logging.getLogger(__name__)

DEFAULT_MESSAGE_LIMIT = 1000
DEFAULT_CHANNEL_TYPES = "public_channel,private_channel"


class SlackAPIWrapper(BaseModel):
    """Wrapper for Slack API."""

    token: Optional[str] = None  # Use SecretStr
    """Token for Slack API"""

    workspace_url: Optional[str] = None
    """URL of the Slack workspace"""

    slack_client: Any = None
    """Client for Slack API"""

    def __init__(self, **kwargs: Any):
        """
        Initialize the SlackAPIWrapper.

        Args:
            token (Optional[str]): Token for Slack API.
            workspace_url (Optional[str]): URL of the Slack workspace.

        Raises:
            ImportError: If the 'slack_sdk' package is not installed.
        """
        kwargs["token"] = get_from_dict_or_env(kwargs, "token", "SLACK_TOKEN", "")
        kwargs["workspace_url"] = get_from_dict_or_env(
            kwargs, "workspace_url", "SLACK_WORKSPACE_URL", ""
        )

        try:
            from slack_sdk import WebClient
        except ImportError:
            raise ImportError(
                "The 'slack_sdk' package is not installed. "
                "Please install it using 'pip install slack_sdk'."
            )
        client = WebClient(token=kwargs["token"])
        kwargs["slack_client"] = client

        super().__init__(**kwargs)

    def get_channel_details_map(
        self, types: Optional[str] = DEFAULT_CHANNEL_TYPES
    ) -> Dict[str, Dict[str, str]]:
        """
        Get a dictionary mapping channel names to their respective details.
        Details include the channel ID, name, and type(public, private).

        Args:
            types (str): Comma-separated list of types of channels to get details from.
                Defaults to "public_channel,private_channel".

        Returns:
            Dict[str, Dict[str, str]]: A dictionary mapping channel names to their
            respective details.
        """
        try:
            response = self.slack_client.conversations_list(types=types)
            channels = response.get("channels", [])
            return {
                channel["name"]: {
                    "id": channel.get("id"),
                    "name": channel.get("name"),
                    "is_private": channel.get("is_private"),
                }
                for channel in channels
            }
        except Exception as e:
            logger.error(f"Error getting channel details map from Slack: {e}")
            return {}

    def get_user_details_map(self) -> Dict[str, Dict[str, str]]:
        """
        Get a dictionary mapping user IDs to their respective details.
        Details include the user id, name, real name, and email.

        Returns:
            Dict[str, Dict[str, str]]: A dictionary mapping user IDs to their
            respective details.
        """
        try:
            response = self.slack_client.users_list()
            users = response.get("members", [])
            return {
                user["id"]: {
                    "name": user.get("name"),
                    "real_name": user.get("real_name"),
                    "email": user.get("profile").get("email"),
                }
                for user in users
            }
        except Exception as e:
            logger.error(f"Error getting user details map from Slack: {e}")
            return {}

    def get_messages(
        self, channel: str, limit: Optional[int] = DEFAULT_MESSAGE_LIMIT
    ) -> Any:
        """
        Get messages from a channel.

        Args:
            channel (str): The channel ID.
            limit (int): The maximum number of messages to return. Defaults to 1000.

        Returns:
            Any: The messages from the channel.
            If message has replies, the replies are included in the message as a list.
            If messages are not available, returns None.
        """
        try:
            response = self.slack_client.conversations_history(
                channel=channel, limit=limit
            )
            messages = response.get("messages", [])

            # Include replies to message in thread if available
            for message in messages:
                if message.get("thread_ts"):
                    thread_response = self.slack_client.conversations_replies(
                        channel=channel, ts=message["ts"]
                    )
                    message["replies"] = thread_response.get("messages", [])
            return messages
        except Exception as e:
            logger.error(f"Error getting messages from Slack: {e}")
            return None

    def get_channel_members(self, channel: str) -> list:
        """
        Get a list of members in a channel.

        Args:
            channel (str): The channel ID.

        Returns:
            list: A list of members in the channel.
        """
        try:
            response = self.slack_client.conversations_members(channel=channel)
            return response.get("members", [])
        except Exception as e:
            logger.error(f"Error getting members for channel {channel}: {e}")
            return []

    def get_authorized_identities(
        self, channel_name: str, user_details_map: dict, channel_details_map: dict
    ) -> list:
        """
        Get a list of authorized identities for a given channel.
        An authorized identity is a user who has access to the channel.
        If the channel is private, only members of the channel are considered
        authorized.
        If the channel is public, all workspace users are considered authorized.

        Args:
            channel_name (str): The channel name.
            user_details_map (dict): A dictionary mapping user IDs to their
                respective details.
            channel_details_map (dict): A dictionary mapping channel names to their
                respective details.

        Returns:
            list: A list of authorized identities(user details) for the given channel.
        """
        try:
            authorized_identities = []

            channel_details = channel_details_map.get(channel_name, {})

            _is_private = channel_details.get("is_private", False)

            if _is_private:
                members = self.get_channel_members(channel_details.get("id"))
            else:
                members = list(user_details_map.keys())

            for member in members:
                user = user_details_map.get(member, {})
                user_email = user.get("email")
                if user_email:
                    authorized_identities.append(user_email)
            return authorized_identities
        except Exception as e:
            logger.error(
                f"Error getting authorized identities for channel {channel_name}: {e}"
            )
            return []
